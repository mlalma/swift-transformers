import Foundation
import MLX

extension ModelUtils {
    /// Maps PyTorch storage class names to MLX data types from given config.
    static func dtype(_ dtypeStr: String?) -> DType? {
        guard let dtypeStr else { return nil }

        switch dtypeStr.lowercased() {
        case "doublestorage", "double", "float64":
            return .float64
        case "floatstorage", "float", "float32":
            return .float32
        case "halfstorage", "float16", "half":
            return .float16
        case "longstorage", "int64", "long":
            return .int64
        case "uint64":
            return .uint64
        case "intstorage", "int", "int32":
            return .int32
        case "shortstorage", "int16", "short":
            return .int16
        case "charstorage", "int8":
            return .int8
        case "bytestorage", "uint8":
            return .uint8
        case "boolstorage", "bool":
            return .bool
        case "bfloat16storage", "bfloat16":
            return .bfloat16
        case "complexdoublestorage", "complex128":
            return nil
        case "completefloatstorage", "complex64":
            return .complex64
        case "quint8storage":
            return .uint8
        case "qint8storage":
            return .int8
        case "qint32storage":
            return .int32
        case "quint4x2storage":
            return .uint8
        case "quint2x4storage":
            return .uint8
        default:
            return nil
        }
    }
}
