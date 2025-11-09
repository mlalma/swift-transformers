import Foundation
import Testing
import MLX

@testable import Hub
@testable import Models

@Suite("PreTrainedConfig tests")
struct PreTrainedConfigTests {
    @Test("Read nanochat configuration")
    func readNanochatConfiguration() async throws {
        let configFile = Bundle.module.url(forResource: "nanochat_config", withExtension: "json")!
        let config = await AutoConfig.fromPretrained(configFile.path())!
        let nanoChatConfig = config as! NanoChatConfig
        
        #expect(nanoChatConfig.attentionBias == false)
        #expect(nanoChatConfig.attentionDropout == 0.0)
        #expect(nanoChatConfig.bosTokenId == 65527)
        #expect(nanoChatConfig.dtype == .bfloat16)
        #expect(nanoChatConfig.eosTokenId == 65531)
        #expect(nanoChatConfig.hiddenAct == "relu2")
        #expect(nanoChatConfig.hiddenSize == 2048)
        #expect(nanoChatConfig.initializerRange.isClose(to: 0.02))
        #expect(nanoChatConfig.intermediateSize == 8192)
        #expect(nanoChatConfig.finalLogitSoftcapping == 15.0)
        #expect(nanoChatConfig.maxPositionEmbeddings == 2048)
        #expect(nanoChatConfig.modelType == "nanochat")
        #expect(nanoChatConfig.numAttentionHeads == 16)
        #expect(nanoChatConfig.numHiddenLayers == 32)
        #expect(nanoChatConfig.numKeyValueHeads == 16)
        #expect(nanoChatConfig.padTokenId == 65531)
        #expect(nanoChatConfig.rmsNormEps.isClose(to: 1e-06))
        #expect(nanoChatConfig.tieWordEmbeddings == false)
        #expect(nanoChatConfig.transformersVersion == "5.0.0.dev0")
        #expect(nanoChatConfig.useCache == true)
        #expect(nanoChatConfig.vocabSize == 65536)
        #expect(nanoChatConfig.ropeParameters?.ropeTheta == 10000.0)
        #expect(nanoChatConfig.ropeParameters?.ropeType == .default)        
    }
}
