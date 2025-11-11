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

    nonisolated(unsafe) static let causalLMNames: [String: () -> PreTrainedModel] = [
        "nanochat": { NanoChatForCausalLM() },
    ]

    nonisolated(unsafe) static let configNames: [String: (Config) -> PreTrainedConfig] = [
        "nanochat": { NanoChatConfig(fromConfig: $0) },
    ]
}
