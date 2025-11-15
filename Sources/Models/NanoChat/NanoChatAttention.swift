import Foundation
import MLX
import MLXNN

/// Multi-headed attention from 'Attention Is All You Need' paper.
final class NanoChatAttention: Module {
    let config: NanoChatConfig
    let layerIdx: Int
    let headDim: Int
    let numKeyValueGroups: Int
    let scaling: Float
    
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    
    let qNorm: NanoChatRMSNorm
    let kNorm: NanoChatRMSNorm
    
    /// Initialize the attention layer.
    /// - Parameters:
    ///   - config: The model configuration.
    ///   - layerIdx: Index of the layer in the model.
    init(config: NanoChatConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        
        // Compute head dimension
        self.headDim = config.hiddenSize / config.numAttentionHeads
        
        // Compute number of key-value groups for GQA
        self.numKeyValueGroups = config.numAttentionHeads / config.numKeyValueHeads
        
        // Compute scaling factor: 1 / sqrt(head_dim)
        self.scaling = pow(Float(headDim), -0.5)
                
        // Initialize projection layers
        let qDim = config.numAttentionHeads * headDim
        let kvDim = config.numKeyValueHeads * headDim
        
        self.qProj = Linear(config.hiddenSize, qDim, bias: config.attentionBias)
        self.kProj = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        self.vProj = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        self.oProj = Linear(qDim, config.hiddenSize, bias: config.attentionBias)
        
        // Initialize RMS normalization layers
        self.qNorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        self.kNorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        
        super.init()
    }
    
    /// Forward pass of the attention layer.
    /// - Parameters:
    ///   - hiddenStates: Input tensor with shape `[batch, seq_len, hidden_size]`.
    ///   - positionEmbeddings: Tuple of (cos, sin) position embeddings from RoPE.
    ///   - attentionMask: Optional attention mask tensor.
    /// - Returns: Tuple of (attention output, attention weights).
    func callAsFunction(
        _ hiddenStates: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray)? = nil,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        
        // Project to query, key, and value
        var queryStates = qProj(hiddenStates)
        var keyStates = kProj(hiddenStates)
        var valueStates = vProj(hiddenStates)
        
        // Reshape and transpose to separate heads
        queryStates = queryStates
            .reshaped(batchSize, seqLen, config.numAttentionHeads, headDim)
            .transposed(axes: [0, 2, 1, 3])
        
        keyStates = keyStates
            .reshaped(batchSize, seqLen, config.numKeyValueHeads, headDim)
            .transposed(axes: [0, 2, 1, 3])
        
        valueStates = valueStates
            .reshaped(batchSize, seqLen, config.numKeyValueHeads, headDim)
            .transposed(axes: [0, 2, 1, 3])
        
        // Apply rotary position embeddings if provided
        if let (cos, sin) = positionEmbeddings {
            (queryStates, keyStates) = applyRotaryPositionEmbedding(
                q: queryStates,
                k: keyStates,
                cos: cos,
                sin: sin
            )
        }
        
        // Apply RMS normalization (RoPE -> Norm, instead of usual Norm -> RoPE)
        queryStates = qNorm(queryStates)
        keyStates = kNorm(keyStates)

        // TODO: Doesn't handle caching, should it?
                
        // Compute attention using eager attention forward
        // TODO: Other attention functions (e.g. flash attention) not yet supported
        let (attnOutput, attnWeights) = eagerAttentionForward(
            query: queryStates,
            key: keyStates,
            value: valueStates,
            attentionMask: attentionMask,
            scaling: scaling,
            numKeyValueGroups: numKeyValueGroups
        )
        
        // Apply output projection
        let outputReshaped = attnOutput.reshaped(batchSize, seqLen, config.numAttentionHeads * headDim)
        let finalOutput = oProj(outputReshaped)
        
        return (finalOutput, attnWeights)
    }
        
    /// Rotates half the hidden dims of the input with flipped signs.
    /// - Parameter x: Input tensor.
    /// - Returns: Rotated tensor with the same shape.
    private func rotateHalf(_ x: MLXArray) -> MLXArray {
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
    private func applyRotaryPositionEmbedding(
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
    /// - Parameters:
    ///   - hiddenStates: The key or value tensor to repeat.
    ///   - numberOfRepetitions: The number of times to repeat each key/value head.
    /// - Returns: The expanded tensor.
    private func repeatKV(hiddenStates: MLXArray, numberOfRepetitions: Int) -> MLXArray {
        guard numberOfRepetitions > 1 else { return hiddenStates }
        
        let batch = hiddenStates.dim(0)
        let numKeyValueHeads = hiddenStates.dim(1)
        let sequenceLength = hiddenStates.dim(2)
        let headDim = hiddenStates.dim(3)
        
        let expanded = hiddenStates.expandedDimensions(axis: 2)
        let repeated = repeated(expanded, count: numberOfRepetitions, axis: 2)
        
        // Reshape to merge dimensions and expand
        return repeated.reshaped(batch, numKeyValueHeads * numberOfRepetitions, sequenceLength, headDim)
    }
    
    /// Eager attention forward pass implementation.
    /// - Parameters:
    ///   - query: Query tensor.
    ///   - key: Key tensor with shape.
    ///   - value: Value tensor with shape.
    ///   - attentionMask: Optional attention mask tensor.
    ///   - scaling: Scaling factor for attention scores, typically 1/sqrt(headDim).
    ///   - numKeyValueGroups: Number of groups for grouped-query attention.
    /// - Returns: Tuple of (attention output, attention weights).
    private func eagerAttentionForward(
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
}
