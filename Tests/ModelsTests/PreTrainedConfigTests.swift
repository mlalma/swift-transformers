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
        #expect(nanoChatConfig.architectures == ["NanoChatForCausalLM"])
    }
    
    @Test(".safetensors model parsing test from local directory")
    func parseModelFromLocalDirectory() async throws {
        let configFile = Bundle.module.url(forResource: "model", withExtension: "safetensors")!
        print(configFile)
        let modelFileDir = configFile.deletingLastPathComponent()
        print("modelFileDir \(modelFileDir)")
        
        ModelMapping.causalLMNames["testmodel"] = { TestModel1() }
        ModelMapping.configNames["testmodel"] = { NanoChatConfig(fromConfig: $0) }
        
        let model = await AutoModelForCausalLM.fromPretrained(modelFileDir.path(), modelArguments: ["use_safetensors": true])
        #expect(model != nil)
        
        // Load model_state_dict.json and compare mappings
        let stateDictFile = Bundle.module.url(forResource: "model_state_dict", withExtension: "json")!
        let stateDictData = try Data(contentsOf: stateDictFile)
        let expectedStateDictJSON = try JSONSerialization.jsonObject(with: stateDictData) as! [String: Any]
        
        // Convert JSON structure to MLXArrays
        var expectedStateDict: [String: MLXArray] = [:]
        for (key, value) in expectedStateDictJSON {
            guard let tensorDict = value as? [String: Any],
                  let shape = tensorDict["shape"] as? [Int],
                  let dtype = tensorDict["dtype"] as? String,
                  let data = tensorDict["data"] as? [Any] else {
                continue
            }
            
            // Convert dtype string to MLX dtype
            let mlxDtype: DType
            switch dtype {
            case "torch.float32":
                mlxDtype = .float32
            case "torch.float16":
                mlxDtype = .float16
            case "torch.bfloat16":
                mlxDtype = .bfloat16
            case "torch.int32":
                mlxDtype = .int32
            case "torch.int64":
                mlxDtype = .int64
            default:
                mlxDtype = .float32
            }
            
            // Create MLXArray from data
            // Flatten multidimensional data recursively
            func flattenData(_ data: Any) -> [Any] {
                if let array = data as? [Any] {
                    return array.flatMap { flattenData($0) }
                } else {
                    return [data]
                }
            }
            
            let flattenedData = flattenData(data)
            
            // Convert data to appropriate Swift array based on dtype
            if mlxDtype == .int32 || mlxDtype == .int64 {
                let intData = flattenedData.compactMap { $0 as? Int }
                expectedStateDict[key] = MLXArray(intData, shape)
            } else {
                // Handle both Int and Double from JSON
                let floatData = flattenedData.compactMap { value -> Float? in
                    if let doubleVal = value as? Double {
                        return Float(doubleVal)
                    } else if let intVal = value as? Int {
                        return Float(intVal)
                    }
                    return nil
                }
                expectedStateDict[key] = MLXArray(floatData, shape)
            }
        }
        
        // Cast model to TestModel1 to access stateDict
        let testModel = model as! TestModel1
        
        // Compare keys
        let expectedKeys = Set(expectedStateDict.keys)
        let actualKeys = Set(testModel.stateDict.keys)
                
        // Check if all expected keys are present in the model
        #expect(expectedKeys == actualKeys, "State dict keys should match")
        
        // Verify each key is present and compare values
        for key in expectedKeys {
            guard let actualArray = testModel.stateDict[key],
                  let expectedArray = expectedStateDict[key] else {
                #expect(Bool(false), "Model should contain key: \(key)")
                continue
            }
            
            // Compare shapes
            #expect(actualArray.shape == expectedArray.shape, 
                   "Shape mismatch for key '\(key)': expected \(expectedArray.shape), got \(actualArray.shape)")
            
            // Compare dtypes
            #expect(actualArray.dtype == expectedArray.dtype,
                   "Dtype mismatch for key '\(key)': expected \(expectedArray.dtype), got \(actualArray.dtype)")
            
            // Compare values - check if arrays are approximately equal
            let isClose = MLX.allClose(actualArray, expectedArray, rtol: 1e-5, atol: 1e-8)
            #expect(isClose.item() == true, 
                   "Value mismatch for key '\(key)': arrays are not close")
        }
        
        // Check for any extra keys in the model that weren't expected
        let extraKeys = actualKeys.subtracting(expectedKeys)
        #expect(extraKeys.isEmpty, "Model should not have extra keys: \(extraKeys)")
        
        // Check for any missing keys in the model
        let missingKeys = expectedKeys.subtracting(actualKeys)
        #expect(missingKeys.isEmpty, "Model should not be missing keys: \(missingKeys)")
    }
    
}
