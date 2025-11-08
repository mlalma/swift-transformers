import Foundation
import Hub

extension Config {
  /// Subscript accessor for dictionary values using String keys with type inference.
  /// - Parameters:
  ///   - key: The string key to look up in the dictionary
  ///   - output: The expected output type (e.g., String.self, Int.self, etc.)
  /// - Returns: The value associated with the key in the requested type, or nil if not found or conversion fails.
  subscript<T>(key: String, _ output: T.Type) -> T? {
    // Check if the Config contains a dictionary
    guard let dict = self.dictionary() else {
      return nil
    }
    
    let binaryKey = BinaryDistinctString(key)
    guard let value = dict[binaryKey] ?? dict[uncamelCase(binaryKey)] else {
      return nil
    }
    
    // Handle different output types
    if output == String.self {
      return value.string() as? T
    } else if output == Int.self {
      return value.integer() as? T
    } else if output == Bool.self {
      return value.boolean() as? T
    } else if output == Double.self {
      return value.floating() as? T
    } else if output == Float.self {
      return value.floating() as? T
    } else if output == Config.self {
      return value as? T
    } else if output == [String].self {
      return value.array()?.compactMap { $0.string() } as? T
    } else if output == [Int].self {
      return value.array()?.compactMap { $0.integer() } as? T
    } else if output == [Double].self {
      return value.array()?.compactMap { $0.floating() as? Double } as? T
    } else if output == [Float].self {
      return value.array()?.compactMap { $0.floating() } as? T
    } else if output == [Config].self {
      return value.array() as? T
    } else if output == [Any].self {
      return value.array()?.compactMap { $0.toAny() } as? T
    } else if output == [String: Any].self {
      if let dictValue = value.dictionary() {
        var result: [String: Any] = [:]
        for (key, val) in dictValue {
          result[key.string] = val.toAny()
        }
        return result as? T
      }
      return nil
    } else if output == [String: Config].self {
      if let dictValue = value.dictionary() {
        var result: [String: Config] = [:]
        for (key, val) in dictValue {
          result[key.string] = val
        }
        return result as? T
      }
      return nil
    }
    
    return nil
  }
  
  /// Simplified subscript that returns Config value (original behavior).
  /// - Parameter key: The string key to look up in the dictionary
  /// - Returns: The Config value associated with the key, or nil if not found
  subscript(key: String) -> Config? {
    guard let dict = self.dictionary() else {
      return nil
    }
    let binaryKey = BinaryDistinctString(key)
    return dict[binaryKey] ?? dict[uncamelCase(binaryKey)]
  }
}

extension Config {
  /// Converts a Config value to Any type.
  ///
  /// - Returns: The appropriate Swift type (String, Int, Bool, Double, Array, Dictionary) or nil
  func toAny() -> Any? {
    if let stringValue = self.string() {
      return stringValue
    } else if let intValue = self.integer() {
      return intValue
    } else if let boolValue = self.boolean() {
      return boolValue
    } else if let floatValue = self.floating() {
      return floatValue
    } else if let arrayValue = self.array() {
      return arrayValue.compactMap { $0.toAny() }
    } else if let dictValue = self.dictionary() {
      var result: [String: Any] = [:]
      for (key, value) in dictValue {
        result[key.string] = value.toAny()
      }
      return result
    } else if self.isNull() {
      return nil
    }
    
    return nil
  }
}
