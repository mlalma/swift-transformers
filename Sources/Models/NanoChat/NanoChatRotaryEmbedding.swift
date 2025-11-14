import Foundation
import MLX
import MLXNN

/// Rotary Position Embedding (RoPE) implementation for NanoChat using MLX.
///
/// This class handles the computation of rotary embeddings for positional encoding
/// in transformer models. It supports various RoPE types including default, linear,
/// dynamic, yarn, longrope, and llama3.
final class NanoChatRotaryEmbedding {
    let config: NanoChatConfig
    
    private(set) var invFreq: MLXArray
    private(set) var originalInvFreq: MLXArray
    private(set) var maxSeqLenCached: Int
    let originalMaxSeqLen: Int
    let ropeType: RopeType
    let attentionScaling: Float
        
    /// Initialize the rotary embedding with a configuration.
    /// - Parameters:
    ///   - config: The NanoChat configuration containing RoPE parameters.
    ///   - device: The device to use for computation (currently MLX manages this automatically).
    public init(config: NanoChatConfig) throws {
        self.config = config
        self.maxSeqLenCached = config.maxPositionEmbeddings
        self.originalMaxSeqLen = config.maxPositionEmbeddings
        
        // Extract rope_type from config
        if let ropeParams = config.ropeParameters {
            self.ropeType = ropeParams.ropeType
        } else {
            self.ropeType = .default
        }
        
        // Compute inverse frequencies based on RoPE type
        let (invFreq, attentionScaling): (MLXArray, Float)
        
        if self.ropeType == .default {
            (invFreq, attentionScaling) = try RopeUtils.computeDefaultRopeParameters(config: config)
        } else {
            // Use the appropriate RoPE initialization function
            (invFreq, attentionScaling) = try RopeUtils.computeRopeParameters(
                config: config,
                ropeType: self.ropeType
            )
        }
        
        self.invFreq = invFreq
        self.originalInvFreq = invFreq
        self.attentionScaling = attentionScaling
    }
        
    /// Compute cosine and sine embeddings for the given input and position IDs.
    ///
    /// - Parameters:
    ///   - x: Input tensor with shape [batch, seq_len, hidden_dim].
    ///   - positionIds: Position indices with shape [batch, seq_len].
    /// - Returns: A tuple of (cos, sin) embeddings with the same dtype as input.
    /*
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        // Expand inv_freq: [None, dim/2, None] -> [batch, dim/2, 1]
        let invFreqExpanded = invFreq
            .reshaped([1, invFreq.shape[0], 1])
            .asType(.float32)
            .broadcast(to: [positionIds.shape[0], invFreq.shape[0], 1])
        
        // Expand position_ids: [batch, None, seq_len]
        let positionIdsExpanded = positionIds
            .reshaped([positionIds.shape[0], 1, positionIds.shape[1]])
            .asType(.float32)
        
        // Compute frequencies: [batch, dim/2, seq_len]
        let freqs = MLX.matmul(invFreqExpanded, positionIdsExpanded).transposed(axes: [0, 2, 1])
        
        // Concatenate frequencies: [batch, seq_len, dim]
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        
        // Compute cos and sin with attention scaling
        let cos = MLX.cos(emb) * attentionScaling
        let sin = MLX.sin(emb) * attentionScaling
        
        // Return with same dtype as input
        return (cos.asType(x.dtype), sin.asType(x.dtype))
    }
    */
    
    
}

// MARK: - Helper Functions

/// Rotates half the hidden dims of the input with flipped signs.
///
/// - Parameter x: Input tensor with shape [..., dim].
/// - Returns: Rotated tensor with the same shape.
public func rotateHalf(_ x: MLXArray) -> MLXArray {
    let dim = x.shape[x.shape.count - 1]
    let halfDim = dim / 2
    
    let x1 = x[.ellipsis, 0..<halfDim]
    let x2 = x[.ellipsis, halfDim..<dim]
    
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Applies Rotary Position Embedding to query and key tensors.
///
/// - Parameters:
///   - q: Query tensor with shape [batch, heads, seq_len, head_dim].
///   - k: Key tensor with shape [batch, heads, seq_len, head_dim].
///   - cos: Cosine embeddings from RoPE.
///   - sin: Sine embeddings from RoPE.
///   - unsqueezeDim: Dimension to unsqueeze cos/sin for broadcasting (default: 1).
/// - Returns: Tuple of (rotated query, rotated key).
public func applyRotaryPosEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    unsqueezeDim: Int = 1
) -> (MLXArray, MLXArray) {
    // Expand dimensions for broadcasting
    var cosExpanded = cos
    var sinExpanded = sin
    
    // Insert dimension at unsqueezeDim
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

