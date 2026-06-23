import Foundation
import NaturalLanguage
struct Item: Codable { let label: String; let text: String }
let items = try! JSONDecoder().decode([Item].self, from: Data(contentsOf: URL(fileURLWithPath: "anti_ai_corpus.json")))
guard let emb = try? NLContextualEmbedding(language: .english) else { FileHandle.standardError.write(Data("no emb\n".utf8)); exit(1) }
try? emb.load()
func vec(_ s: String) -> [Double]? {
    guard let r = try? emb.embeddingResult(for: s, language: .english) else { return nil }
    var sum:[Double]=[]; var n=0
    r.enumerateTokenVectors(in: s.startIndex..<s.endIndex){ v,_ in
        if sum.isEmpty { sum=[Double](repeating:0,count:v.count) }
        for i in 0..<v.count { sum[i]+=Double(v[i]) }; n+=1; return true }
    return n==0 ? nil : sum.map{ $0/Double(n) }
}
var out:[[String:Any]]=[]
for it in items { if let v=vec(it.text) { out.append(["label":it.label,"text":it.text,"v":v]) } }
let d = try! JSONSerialization.data(withJSONObject: out)
try! d.write(to: URL(fileURLWithPath: "corpus_emb.json"))
print("embedded \(out.count)/\(items.count) -> corpus_emb.json")
