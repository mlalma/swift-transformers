import Foundation
import MLX
import MLXNN

/// Rotary Position Embedding (RoPE) implementation for NanoChat.
final class NanoChatRotaryEmbedding : Module {
    let config: NanoChatConfig
    
    private(set) var invFreq: MLXArray
    private(set) var originalInvFreq: MLXArray
    private(set) var maxSeqLenCached: Int
    let originalMaxSeqLen: Int
    let ropeType: RopeType
    let attentionScaling: Float
        
    /// Initialize the rotary embedding with a configuration.
    /// - Parameter config: The NanoChat configuration containing RoPE parameters.
    init(config: NanoChatConfig) throws {
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
        let (invFreq, attentionScaling) =
            try RopeUtils.computeRopeParameters(
                config: config,
                ropeType: self.ropeType)
                
        self.invFreq = invFreq
        self.originalInvFreq = invFreq
        self.attentionScaling = attentionScaling
    }
        
    /// Compute cosine and sine embeddings for the given input and position IDs.
    /// - Parameters:
    ///   - x: Input tensor with shape [batch, seq_len, hidden_dim].
    ///   - positionIds: Position indices with shape [batch, seq_len].
    /// - Returns: A tuple of (cos, sin) embeddings with the same dtype as input.
    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        // Shape: [1, inv_freq_size, 1]
        let invFreqExpanded = invFreq
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 2)
            .asType(.float32)

        // Shape: [batch_size, 1, seq_len]
        let positionIdsExpanded = positionIds
            .expandedDimensions(axis: 1)
            .asType(.float32)
        
        // Compute frequencies: [batch, dim/2, seq_len]
        // Let MLX handle broadcasting
        let freqs = MLX.matmul(invFreqExpanded, positionIdsExpanded).transposed(axes: [0, 2, 1])
        
        // Concatenate frequencies: [batch, seq_len, dim]
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        
        let cos = MLX.cos(emb) * attentionScaling
        let sin = MLX.sin(emb) * attentionScaling
        return (cos.asType(x.dtype), sin.asType(x.dtype))
    }
}

