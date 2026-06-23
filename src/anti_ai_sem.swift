#!/usr/bin/env swift
// Embeds the turn, scores its margin toward AI vs human examples in corpus_emb.json.
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
// if we already blocked once this turn, don't block again — avoids the re-block loop
if (obj["stop_hook_active"] as? Bool) == true { allow("stop_hook_active") }

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

// Contrastive margin: how much closer is this passage to the AI examples than to
// the human ones. A signed reason, not a bare vote. Skip exact-text self matches.
var aiSims: [Double] = [], huSims: [Double] = []
for r in refs where r.text != text {
    let c = cos(q, r.v)
    if r.label == "ai" { aiSims.append(c) } else { huSims.append(c) }
}
func topMean(_ a: [Double], _ k: Int) -> Double {
    let s = a.sorted(by: >).prefix(k); return s.isEmpty ? 0 : s.reduce(0, +) / Double(s.count)
}
let aiMean = topMean(aiSims, 5), huMean = topMean(huSims, 5)
let margin = aiMean - huMean
let MARGIN: Double = {
    if let s = ProcessInfo.processInfo.environment["SR_MARGIN"], let v = Double(s) { return v }
    return 0.02
}()

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

// Positional nudge: mean-pooling discards word order, so add it back as a small,
// bounded adjustment. The embedding `margin` stays the base of the score.
let opening = String(text.prefix(120)).lowercased()
let openerPraise = rx("^\\W*(great|excellent|fantastic|brilliant|wonderful|amazing|perfect)\\s+(question|point|observation)") || opening.range(of: #"^\W*(honestly|frankly|absolutely|certainly|of course)\b"#, options: .regularExpression) != nil
func anyRx(_ p: String) -> Bool { lower.range(of: p, options: .regularExpression) != nil }
let validation = anyRx(#"your (theory|hypothesis|framing|instinct|intuition|perspective|premise) (is|sounds)|you raise (a )?(great|good|valid|important) point|completely (valid|understandable)|spot on|on the right track|you'?re (absolutely )?right"#)
// only negation / self-correction markers — substance markers like "on line N"
// or "I don't know" are too easy to prepend to slop, so they're excluded.
let disagree = anyRx(#"\bhowever\b|\bin fact\b|not (quite|the case|true|wrong|right|the|outside|inside)\b|i'?d push back|the (data|evidence) (does|doesn'?t|suggest|show)|the opposite|but no\b|but not\b|but the\b|i misread|absolutely not|absolutely no\b"#)
var nudge = 0.0, why: [String] = []
if openerPraise { nudge += 0.05; why.append("opener-praise") }
if validation   { nudge += 0.03; why.append("validation") }
if disagree     { nudge -= 0.06; why.append("disagreement(redeems)") }
let score = margin + nudge

// verdict: embedding margin + bounded positional nudge. borderline passages pass.
if score > MARGIN {
    block(String(format: "score=%.2f (margin=%.2f, nudge=%+.2f %@); evidence: %@",
                 score, margin, nudge, why.isEmpty ? "—" : why.joined(separator: ","),
                 struc.isEmpty ? "none" : struc.joined(separator: ",")))
}
allow(String(format: "score=%.2f (margin=%.2f nudge=%+.2f) — reads human", score, margin, nudge))
