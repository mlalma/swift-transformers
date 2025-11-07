//
//  ModelMapping.swift
//  swift-transformers
//
//  Created by Lassi Maksimainen on 07.11.2025.
//

enum ModelMapping {
}

extension ModelMapping {
  static let names: [String: String] = [
    "nanochat": "NanoChatModel"
  ]

  static let pretrainingNames: [String: String] = [
    "nanochat": "NanoChatForCausalLM"
  ]

  static let causalLMNames: [String: String] = [
    "nanochat": "NanoChatForCausalLM"
  ]
}
