#!/usr/bin/env swift
// Embeds the turn and does a k-NN vote against corpus_emb.json.
// exit 2 = flagged, 0 = not. Short or broken input exits 0.
import Foundation
import NaturalLanguage

func allow(_ why: String = "") -> Never {
    if !why.isEmpty { FileHandle.standardError.write(Data("[anti-ai-sem] allow: \(why)\n".utf8)) }
    exit(0)
}
func block(_ why: String) -> Never {
    FileHandle.standardError.write(Data("BLOCKED by anti-ai-sem (reads as AI-generated): \(why)\n".utf8))
    exit(2)
}

let dir = URL(fileURLWithPath: CommandLine.arguments.first ?? "").deletingLastPathComponent()
func path(_ n: String) -> String {
    // standalone layout: bin/<binary> with data in ../corpus/. Try a few spots.
    let candidates = [
        dir.appendingPathComponent(n).path,
        dir.appendingPathComponent("../corpus/\(n)").path,
        NSString(string: "~/semantic-regex/anti-ai-writing/corpus/\(n)").expandingTildeInPath,
    ]
    return candidates.first { FileManager.default.fileExists(atPath: $0) } ?? candidates[0]
}

struct Ref { let label: String; let text: String; let v: [Double] }
let refs: [Ref] = {
    guard let d = try? Data(contentsOf: URL(fileURLWithPath: path("corpus_emb.json"))),
          let arr = (try? JSONSerialization.jsonObject(with: d)) as? [[String: Any]] else { return [] }
    return arr.compactMap { o in
        guard let l = o["label"] as? String, let t = o["text"] as? String, let v = o["v"] as? [Double] else { return nil }
        return Ref(label: l, text: t, v: v)
    }
}()
if refs.isEmpty { allow("no reference embeddings") }

let input = FileHandle.standardInput.readDataToEndOfFile()
if input.isEmpty { allow("empty stdin") }
guard let obj = (try? JSONSerialization.jsonObject(with: input)) as? [String: Any] else { allow("invalid stdin") }

func lastAssistantText(_ p: String) -> String {
    guard let c = try? String(contentsOfFile: p, encoding: .utf8) else { return "" }
    var last = ""
    for line in c.split(separator: "\n") {
        guard let d = line.data(using: .utf8), let o = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { continue }
        let m = o["message"] as? [String: Any]
        guard (o["type"] as? String) == "assistant" || (m?["role"] as? String) == "assistant" else { continue }
        if let s = m?["content"] as? String { last = s }
        else if let a = m?["content"] as? [Any] {
            let t = a.compactMap { ($0 as? [String: Any]).flatMap { ($0["type"] as? String) == "text" ? $0["text"] as? String : nil } }.joined(separator: " ")
            if !t.isEmpty { last = t }
        }
    }
    return last
}

var text = ""
if let tp = obj["transcript_path"] as? String { text = lastAssistantText(tp) }
if text.isEmpty, let t = obj["_text"] as? String { text = t }
text = text.trimmingCharacters(in: .whitespacesAndNewlines)
if text.isEmpty { allow("empty turn") }
if text.split(whereSeparator: { $0 == " " || $0 == "\n" }).count < 6 { allow("too short to judge style") }

guard let emb = try? NLContextualEmbedding(language: .english) else { allow("no embedder") }
try? emb.load()
func vec(_ s: String) -> [Double]? {
    guard let r = try? emb.embeddingResult(for: s, language: .english) else { return nil }
    var sum: [Double] = []; var n = 0
    r.enumerateTokenVectors(in: s.startIndex..<s.endIndex) { v, _ in
        if sum.isEmpty { sum = [Double](repeating: 0, count: v.count) }
        for i in 0..<v.count { sum[i] += Double(v[i]) }; n += 1; return true
    }
    return n == 0 ? nil : sum.map { $0 / Double(n) }
}
guard let q = vec(text) else { allow("could not embed turn") }
func cos(_ a: [Double], _ b: [Double]) -> Double {
    var d = 0.0, na = 0.0, nb = 0.0
    for i in 0..<min(a.count, b.count) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return (na == 0 || nb == 0) ? 0 : d / (na.squareRoot()*nb.squareRoot())
}

// k-NN, skipping any exact-text match so it can't vote for itself
var sims: [(Double, String)] = []
for r in refs where r.text != text { sims.append((cos(q, r.v), r.label)) }
sims.sort { $0.0 > $1.0 }
let k = 5
let top = sims.prefix(k)
let aiVotes = top.filter { $0.1 == "ai" }.count

// Structural detectors — a passage-level corroborator (never a single word).
func rx(_ p: String) -> Bool { (try? NSRegularExpression(pattern: p, options: [.caseInsensitive]))
    .map { $0.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil } ?? false }
let lower = text.lowercased()
let formulaic = ["in today'?s","more than ever","it'?s worth noting","it is worth noting",
 "let me break (this|it) down","let'?s dive (in|right in)","when it comes to","the truth is",
 "let me (unpack|walk you through)","unlock the (potential|power)","harness the power","a testament to",
 "thrilled to (announce|share)","i'?m (thrilled|excited|proud) to","fast[- ]paced","the power of",
 "embark on","navigate the complexities","furthermore","moreover","on a mission to","take your"]
var struc: [String] = []
if text.contains("—") || rx("\\s-\\s") { struc.append("em_dash") }
if rx("\\b\\w+,\\s+\\w+,\\s+(and |or )?\\w+") { struc.append("rule_of_three") }
if rx("not just .* but|it'?s not about .* it'?s about|it'?s not .*,\\s*it'?s") { struc.append("not_just_but") }
if formulaic.contains(where: { rx($0) }) { struc.append("formulaic_transition") }
if rx("^\\W*(honestly|frankly|absolutely|certainly|great question|i'?d be happy|happy to|of course|that'?s a (great|fantastic) (question|point))")
   && rx("\\b(great|fantastic|brilliant|excellent|amazing|incredible|insightful|right|nail on the head)\\b") {
    struc.append("hedge_praise_opener")
}
// >=3 distinct slop stems = a density signal (evidence only)
let stems = ["delve","leverag","utiliz","facilitat","underscor","foster","streamlin","elevat",
 "empower","unlock","unleash","harness","showcas","spearhead","cultivat","robust","seamless",
 "comprehensiv","intricat","pivotal","crucial","holistic","nuanced","multifacet","transformativ",
 "groundbreak","revolution","innovati","paramount","vibrant","captivat","compelling","meticulous",
 "tapestry","realm","landscape","synerg","testament","cornerstone","paradigm","ecosystem",
 "cutting-edge","navigat"]
let toks = Set(lower.split(whereSeparator: { !($0.isLetter || $0 == "-") }).map(String.init))
let hits = stems.filter { s in toks.contains(where: { $0.hasPrefix(s) }) }
if hits.count >= 3 { struc.append("buzzword_stacking") }

// verdict is the k-NN vote only; structural list is just logged as evidence
let nn = top.first.map { String(format: "cos=%.2f", $0.0) } ?? ""
if aiVotes >= 3 {
    block("semantic: \(aiVotes)/\(k) nearest AI (\(nn)); structural evidence: \(struc.isEmpty ? "none" : struc.joined(separator: ","))")
}
allow("aiVotes=\(aiVotes)/\(k) structural=\(struc) — reads human")
