import Foundation
import NaturalLanguage
let text = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? ""
guard let emb = try? NLContextualEmbedding(language: .english) else { print("[]"); exit(0) }
try? emb.load()
guard let r = try? emb.embeddingResult(for: text, language: .english) else { print("[]"); exit(0) }
var sum: [Double] = []; var n = 0
r.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { v, _ in
    if sum.isEmpty { sum = [Double](repeating: 0, count: v.count) }
    for i in 0..<v.count { sum[i] += Double(v[i]) }; n += 1; return true
}
let mean = n == 0 ? [] : sum.map { $0 / Double(n) }
print(String(data: try! JSONSerialization.data(withJSONObject: mean), encoding: .utf8)!)
