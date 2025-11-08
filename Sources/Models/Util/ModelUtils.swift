import Foundation

enum ModelUtils {
  static let version = "5.0.0-dev.0"
}

extension ModelUtils {
  static func isRemoteURL(_ url: String) -> Bool {
    guard let urlComponents = URLComponents(string: url),
          let scheme = urlComponents.scheme?.lowercased() else {
      return false
    }
        
    let remoteSchemes = ["http", "https"]
    return remoteSchemes.contains(scheme) && urlComponents.host != nil
  }
}
