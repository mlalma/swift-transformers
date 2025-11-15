import Foundation
import MLX
import MLXNN

/// Rotates half the hidden dims of the input with flipped signs.
/// - Parameter x: Input tensor.
/// - Returns: Rotated tensor with the same shape.
func rotateHalf(_ x: MLXArray) -> MLXArray {
    let dim = x.shape[x.shape.count - 1]
    let halfDim = dim / 2
    
    let x1 = x[.ellipsis, 0..<halfDim]
    let x2 = x[.ellipsis, halfDim..<dim]
    
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Applies Rotary Position Embedding to query and key tensors.
/// - Parameters:
///   - q: Query tensor with shape [batch, heads, seq_len, head_dim].
///   - k: Key tensor with shape [batch, heads, seq_len, head_dim].
///   - cos: Cosine embeddings from RoPE.
///   - sin: Sine embeddings from RoPE.
///   - unsqueezeDim: Dimension to unsqueeze cos/sin for broadcasting (default: 1).
/// - Returns: Tuple of (rotated query, rotated key).
func applyRotaryPositionEmbedding(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    unsqueezeDim: Int = 1
) -> (MLXArray, MLXArray) {
    var cosExpanded = cos
    var sinExpanded = sin
    
    var cosShape = cos.shape
    cosShape.insert(1, at: unsqueezeDim)
    cosExpanded = cos.reshaped(cosShape)
    
    var sinShape = sin.shape
    sinShape.insert(1, at: unsqueezeDim)
    sinExpanded = sin.reshaped(sinShape)
    
    // Apply rotary embeddings
    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
    
    return (qEmbed, kEmbed)
}

/// Repeats key/value heads to match the number of query heads for grouped-query attention (GQA).
/// This function enables efficient grouped-query attention by repeating each key/value head
/// multiple times to match the number of query heads.
/// - Parameters:
///   - hiddenStates: The key or value tensor to repeat, with shape
///                   `[batch, num_key_value_heads, seq_len, head_dim]`.
///                   This represents the grouped key/value heads that need to be
///                   expanded to match the number of query heads.
///   - numberOfRepetitions: The number of times to repeat each key/value head.
///                          This is calculated as `num_query_heads / num_key_value_heads`.
///                          For standard multi-head attention, this is 1 (no repetition needed).
///                          For multi-query attention (MQA), this equals `num_query_heads`.
///                          For grouped-query attention (GQA), this is the group size.
/// - Returns: The expanded tensor with shape `[batch, num_attention_heads, seq_len, head_dim]`,
///            where each key/value head has been repeated `numberOfRepetitions` times along the head dimension.
func repeatKV(hiddenStates: MLXArray, numberOfRepetitions: Int) -> MLXArray {
    guard numberOfRepetitions > 1 else { return hiddenStates }

    let batch = hiddenStates.dim(0)
    let numKeyValueHeads = hiddenStates.dim(1)
    let sequenceLength = hiddenStates.dim(2)
    let headDim = hiddenStates.dim(3)

    let expanded = hiddenStates.expandedDimensions(axis: 2)
    let repeated = repeated(expanded, count: numberOfRepetitions, axis: 2)

    // Reshape to merge dimensions and expand
    // (batch, num_key_value_heads * numberOfRepetitions, sequenceLength, headDim)
    return repeated.reshaped(batch, numKeyValueHeads * numberOfRepetitions, sequenceLength, headDim)
}

/// Eager attention forward pass implementation.
/// - Parameters:
///   - query: Query tensor.
///   - key: Key tensor.
///   - value: Value tensor.
///   - attentionMask: Optional attention mask tensor.
///   - scaling: Scaling factor for attention scores, typically 1/sqrt(headDim).
///   - numKeyValueGroups: Number of groups for grouped-query attention.
/// - Returns: Tuple of (attention output, attention weights).
func eagerAttentionForward(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    attentionMask: MLXArray?,
    scaling: Float,
    numKeyValueGroups: Int
) -> (MLXArray, MLXArray) {
    // Repeat key and value tensors for grouped-query attention
    let keyStates = repeatKV(hiddenStates: key, numberOfRepetitions: numKeyValueGroups)
    let valueStates = repeatKV(hiddenStates: value, numberOfRepetitions: numKeyValueGroups)
    
    // Compute attention scores
    let keyTransposed = keyStates.transposed(axes: [0, 1, 3, 2])
    var attnWeights = MLX.matmul(query, keyTransposed) * scaling
    
    // Apply attention mask if provided
    if let mask = attentionMask {
        let keySeqLen = keyStates.shape[2]
        let causalMask = mask[.ellipsis, .newAxis, .newAxis, 0..<keySeqLen]
        attnWeights = attnWeights + causalMask
    }
    
    // Apply softmax
    let originalDtype = query.dtype
    attnWeights = MLX.softmax(attnWeights.asType(.float32), axis: -1).asType(originalDtype)
        
    // Compute attention output
    var attnOutput = MLX.matmul(attnWeights, valueStates)
    
    // Transpose for output projection
    attnOutput = attnOutput.transposed(axes: [0, 2, 1, 3])
    
    return (attnOutput, attnWeights)
}
