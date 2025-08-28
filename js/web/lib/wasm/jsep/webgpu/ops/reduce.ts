// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramShaderCacheInfo } from '../types';

import { createTensorShapeVariables, IndicesHelper, inputVariable, outputVariable, ShaderHelper } from './common';
import {
  reduceL1Shared,
  reduceL2Shared,
  reduceLogSumExpShared,
  reduceLogSumShared,
  reduceMaxShared,
  reduceMeanShared,
  reduceMinShared,
  reduceProdShared,
  reduceSumShared,
  reduceSumSquareShared,
} from './reduce-shared';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('Reduce op requires 1 or 2 inputs.');
  }

  if (inputs.length === 2 && inputs[1].dims.length !== 1) {
    throw new Error('Invalid axes input dims.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

export type ReduceOp = (
  input: IndicesHelper,
  output: IndicesHelper,
  axes: readonly number[],
) => [string, string, string, string, ...string[]];

const noOp: ReduceOp = (input) => ['', '', `var value = ${input.getByIndices('input_indices')};`, ''];
export const createReduceProgramInfo = (
  name: string,
  shaderCache: ProgramShaderCacheInfo,
  inputs: readonly TensorView[],
  reduceOp: ReduceOp,
  axesInput: number[],
  outputDataType: DataType,
  keepDims = false,
  noopWithEmptyAxes = false,
): ProgramInfo => {
  const outputShape: number[] = [];
  const inputShape = inputs[0].dims;
  const inputRank = inputShape.length;
  const axes = ShapeUtil.normalizeAxes(axesInput, inputRank);
  const reduceOnAllAxes = !noopWithEmptyAxes && axes.length === 0;
  inputShape.forEach((d, i) => {
    if (reduceOnAllAxes || axes.indexOf(i) >= 0) {
      if (keepDims) {
        outputShape.push(1);
      } // else { // skip this axis}
    } else {
      outputShape.push(d);
    }
  });
  const outputRank = outputShape.length;
  const outputSize = ShapeUtil.size(outputShape);
  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const idxCopy: string[] = []; // copy output indexes to input indexes

    const input = inputVariable('_A', inputs[0].dataType, inputRank);
    const output = outputVariable('output', outputDataType, outputRank);
    const ops = reduceOp(input, output, axes);
    let reduceOps = ops[2];

    for (let k = 0, l = 0; k < inputRank; k++) {
      // if this axis is reduced
      if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
        if (keepDims) {
          l++;
        }
        // loop over the d-th axis
        reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputShape[k]}; j${k}++) {
                  ${ops[2].includes('last_index') ? `let last_index = j${k};` : ''}
                  ${input.indicesSet('input_indices', k, `j${k}`)}
                  ${reduceOps}
                }`;
      } else {
        idxCopy.push(`${input.indicesSet('input_indices', k, output.indicesGet('output_indices', l))};`);
        l++;
      }
    }
    return `

        ${shaderHelper.registerUniform('output_size', 'u32').declareVariables(input, output)}

        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
          var input_indices: ${input.type.indices};
          let output_indices = ${output.offsetToIndices('global_idx')};

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${ops[1]}
          ${reduceOps}
          ${ops[3]}
          ${ops.length === 4 ? output.setByOffset('global_idx', 'value') : ops.slice(4).join('\n')}
        }`;
  };

  return {
    name,
    shaderCache,
    getShaderSource,
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType: outputDataType }],
      dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
      programUniforms: [
        { type: DataType.uint32, data: outputSize },
        ...createTensorShapeVariables(inputShape, outputShape),
      ],
    }),
  };
};

export const createReduceAttributesFromInputs = (
  inputs: readonly TensorView[],
  attributes: ReduceAttributes,
): ReduceAttributes => {
  const axes: number[] = [];
  if (inputs[1].dims[0] > 0) {
    inputs[1].getBigInt64Array().forEach((v) => axes.push(Number(v)));
  }
  return createAttributeWithCacheKey({
    axes,
    keepDims: attributes.keepDims,
    noopWithEmptyAxes: attributes.noopWithEmptyAxes,
  });
};

const runReduceProgram = async (
  context: ComputeContext,
  name: string,
  attributes: ReduceAttributes,
  reduceOp: ReduceOp,
): Promise<void> => {
  const inputs = context.inputs;
  const updatedAttributes: ReduceAttributes =
    inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);

  await context.compute(
    createReduceProgramInfo(
      name,
      { hint: updatedAttributes.cacheKey, inputDependencies: ['rank'] },
      [inputs[0]],
      updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp,
      updatedAttributes.axes,
      inputs[0].dataType,
      updatedAttributes.keepDims,
      updatedAttributes.noopWithEmptyAxes,
    ),
    { inputs: [0] },
  );
};

const reduceLogSumNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var value = ${output.type.storage}(0);`,
    '',
    `value += ${input.getByIndices('input_indices')};`,
    'value = log(value);',
  ];
  await runReduceProgram(context, 'ReduceLogSum', attributes, reduceOp);
};

const reduceL1Naive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var value = ${output.type.storage}(0);`,
    '',
    `value += abs(${input.getByIndices('input_indices')});`,
    '',
  ];
  await runReduceProgram(context, 'ReduceL1', attributes, reduceOp);
};

const reduceL2Naive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var t = ${output.type.value}(0); var value = ${output.type.value}(0);`,
    '',
    `t = ${input.getByIndices('input_indices')}; value += (t * t);`,
    'value = sqrt(value);',
  ];
  await runReduceProgram(context, 'ReduceL2', attributes, reduceOp);
};

const reduceLogSumExpNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var value = ${output.type.storage}(0);`,
    '',
    `value += exp(${input.getByIndices('input_indices')});`,
    'value = log(value);',
  ];
  await runReduceProgram(context, 'ReduceLogSumExp', attributes, reduceOp);
};

const reduceMaxNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, _output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(input.indicesSet('input_indices', k, 0));
      }
    }

    return [
      `${idxZero.join('\n')}`,
      `var value = ${input.getByIndices('input_indices')};`,
      `value = max(value, ${input.getByIndices('input_indices')});`,
      '',
    ];
  };
  await runReduceProgram(context, 'ReduceMax', attributes, reduceOp);
};

const reduceMeanNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output, axes) => {
    let size = 1.0;
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        // TODO: this depends on the input dims. If we want to use uniform, this need to be updated.
        size *= context.inputs[0].dims[k];
      }
    }

    return [
      'var sum = f32(0);',
      '',
      `sum += f32(${input.getByIndices('input_indices')});`,
      `let value = ${output.type.value}(sum / ${size});`,
    ];
  };
  await runReduceProgram(context, 'ReduceMean', attributes, reduceOp);
};

const reduceMinNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, _output, axes) => {
    const idxZero = [];
    for (let k = 0; k < input.rank; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`input_indices[${k}] = 0;`); // first element
      }
    }

    return [
      `${idxZero.join('\n')}`,
      `var value = ${input.getByIndices('input_indices')};`,
      `value = min(value, ${input.getByIndices('input_indices')});`,
      '',
    ];
  };
  await runReduceProgram(context, 'ReduceMin', attributes, reduceOp);
};

const reduceProdNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var value = ${output.type.storage}(1);`,
    '',
    `value *= ${input.getByIndices('input_indices')};`,
    '',
  ];
  await runReduceProgram(context, 'ReduceProd', attributes, reduceOp);
};

const reduceSumNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var value = ${output.type.storage}(0);`,
    '',
    `value += ${input.getByIndices('input_indices')};`,
    '',
  ];
  await runReduceProgram(context, 'ReduceSum', attributes, reduceOp);
};

const reduceSumSquareNaive = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (input, output) => [
    `var t = ${output.type.value}(0); var value = ${output.type.value}(0);`,
    '',
    `t = ${input.getByIndices('input_indices')}; value += t * t;`,
    '',
  ];
  await runReduceProgram(context, 'ReduceSumSquare', attributes, reduceOp);
};

const useNaiveReduceMethod = (
  shape: readonly number[],
  axes: readonly number[],
  noopWithEmptyAxes: boolean,
): boolean => {
  if (axes.length === 0) {
    return noopWithEmptyAxes;
  }

  let outputSize = 1;
  let reduceSize = 1;
  for (let dim = 0; dim < axes.length; dim++) {
    if (axes.indexOf(dim) === -1) {
      outputSize *= shape[dim];
    } else {
      reduceSize *= shape[dim];
    }
  }

  // The condition data is very rough, although considering the count of Execution Unit (EU), the potential
  // work groups in a EU and the counts of loops in the naive and shared methods, also doing experiments
  // on some machines.
  return reduceSize < 32 && outputSize > 1024;
};

export const reduceMean = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceMeanNaive(context, attributes);
  } else {
    await reduceMeanShared(context, attributes);
  }
};

export const reduceL1 = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceL1Naive(context, attributes);
  } else {
    await reduceL1Shared(context, attributes);
  }
};

export const reduceL2 = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceL2Naive(context, attributes);
  } else {
    await reduceL2Shared(context, attributes);
  }
};

export const reduceLogSumExp = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceLogSumExpNaive(context, attributes);
  } else {
    await reduceLogSumExpShared(context, attributes);
  }
};

export const reduceMax = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceMaxNaive(context, attributes);
  } else {
    await reduceMaxShared(context, attributes);
  }
};

export const reduceMin = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceMinNaive(context, attributes);
  } else {
    await reduceMinShared(context, attributes);
  }
};

export const reduceProd = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceProdNaive(context, attributes);
  } else {
    await reduceProdShared(context, attributes);
  }
};

export const reduceSum = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceSumNaive(context, attributes);
  } else {
    await reduceSumShared(context, attributes);
  }
};

export const reduceSumSquare = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceSumSquareNaive(context, attributes);
  } else {
    await reduceSumSquareShared(context, attributes);
  }
};

export const reduceLogSum = async (context: ComputeContext, attributes: ReduceAttributes): Promise<void> => {
  if (useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes)) {
    await reduceLogSumNaive(context, attributes);
  } else {
    await reduceLogSumShared(context, attributes);
  }
};
