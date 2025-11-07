import Foundation
import Hub

class AutoConfig {
  private init() {}
  
  static func forModel(type: String, configArguments: [String: Any]) -> Config? {
    if ModelMapping.configNames.keys.contains(type) {
      // Convert [String: Any] to [NSString: Any] and create Config
      let nsStringDict = configArguments.reduce(into: [NSString: Any]()) { result, element in
        result[element.key as NSString] = element.value
      }
      return Config(nsStringDict)
    }
    return nil
  }
}
