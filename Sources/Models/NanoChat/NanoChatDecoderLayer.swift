import Foundation
import MLX
import MLXNN

/// Decoder layer for the NanoChat model.
/// Self-attention followed by a feed-forward with residual connections and RMS normalization.
final class NanoChatDecoderLayer: Module {
    let hiddenSize: Int
    let selfAttn: NanoChatAttention
    let mlp: NanoChatMLP
    let inputLayernorm: NanoChatRMSNorm
    let postAttentionLayernorm: NanoChatRMSNorm
    
    /// Initialize the decoder layer.
    /// - Parameters:
    ///   - config: Configuration for the NanoChat model.
    ///   - layerIdx: Index of this layer in the model.
    ///   - weights: Weights to use for initialization
    init(config: NanoChatConfig, layerIdx: Int, weights: [String: MLXArray]) throws {
        self.hiddenSize = config.hiddenSize
        
        // Initialize self-attention layer
        self.selfAttn = try NanoChatAttention(config: config, layerIdx: layerIdx, weights: weights)
        
        // Initialize MLP (feed-forward) layer
        self.mlp = try NanoChatMLP(config: config, layerIdx: layerIdx, weights: weights)
        
        // Initialize RMS normalization layers
        self.inputLayernorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        self.postAttentionLayernorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        
        super.init()
    }
    
    /// Forward pass through the decoder layer.
    /// - Parameters:
    ///   - hiddenStates: Input tensor with shape `[batch, seq_len, hidden_size]`.
    ///   - attentionMask: Optional attention mask tensor.
    ///   - positionEmbeddings: Optional tuple of (cos, sin) position embeddings from RoPE.
    /// - Returns: Output tensor after attention and MLP with residual connections.
    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        positionEmbeddings: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        // Self-Attention with pre-normalization and residual connection
        var residual = hiddenStates
        var hiddenStates = inputLayernorm(hiddenStates)
        
        // Apply self-attention (ignore attention weights)
        (hiddenStates, _) = selfAttn(
            hiddenStates,
            positionEmbeddings: positionEmbeddings,
            attentionMask: attentionMask
        )
        
        hiddenStates = residual + hiddenStates
        
        // MLP (Feed-Forward) with pre-normalization and residual connection
        residual = hiddenStates
        hiddenStates = postAttentionLayernorm(hiddenStates)
        hiddenStates = mlp(hiddenStates)
        hiddenStates = residual + hiddenStates
        
        return hiddenStates
    }
}
