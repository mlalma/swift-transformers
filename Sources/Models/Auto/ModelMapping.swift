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

    nonisolated(unsafe) static var causalLMNames: [String: () -> PreTrainedModel] = [
        "nanochat": { NanoChatForCausalLM() },
    ]

    nonisolated(unsafe) static var configNames: [String: (Config) -> PreTrainedConfig] = [
        "nanochat": { NanoChatConfig(fromConfig: $0) },
    ]
}
