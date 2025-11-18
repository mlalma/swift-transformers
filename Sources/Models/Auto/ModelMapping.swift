import Foundation
import Hub

enum ModelMapping {}

extension ModelMapping {
    static let names: [String: String] = [
        "nanochat": "NanoChatModel",
    ]

    static let pretrainingNames: [String: String] = [
        "nanochat": "NanoChatForCausalLM",
    ]

    nonisolated(unsafe) static var causalLMNames: [String: (PreTrainedConfig) throws -> PreTrainedModel] = [
        "nanochat": { try NanoChatForCausalLM(fromConfig: $0 as? NanoChatConfig) },
    ]

    nonisolated(unsafe) static var configNames: [String: (Config) -> PreTrainedConfig] = [
        "nanochat": { NanoChatConfig(fromConfig: $0) },
    ]
}
