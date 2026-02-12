# Milestone: Regex JIT

This was originally the first milestone, but it's waaay too broad. Better to scope back down to a TableGen replacement for now.

## Context

Start with a demonstration dialect that has some meaningful use of JIT compilation.

A good option is regex.
- [Performance report on PCRE (2014)](https://zherczeg.github.io/sljit/pcre.html)
suggests that there's large performance gains for regex computation with a JIT, but the JIT compile cost is high enough to not use it by default in most cases.
- [Combining MLIR Dialects with Domain-Specific Architecture for Efficient Regular Expression Matching](https://dl.acm.org/doi/epdf/10.1145/3696443.3708916) demonstrates an approach for compiling performant regexes from MLIR

A great outcome would be
- target some subset of PCRE
- demonstrate regex performance pcre-jit <= dgen <= prce (no jit)
- demonstrate regex compile speed pcre (no jit) <= dgen <= prce-jit

This would place DGEN on the _pareto frontier_ for this problem. While achieving pareto dominance over PCRE is a non-goal, it would establish the value of the approach.

## Start with a small PCRE Subset

Intentionally minimal to get end-to-end fast:
- Character literals: `a`, `b`, `1`, `.` (any char)
- Quantifiers: `*` (0+), `+` (1+), `?` (0 or 1)
- Alternation: `|`
- Grouping: `(...)`

**Out of scope**: character classes, anchors, escapes, `\d`/`\w`/`\s`, `{n,m}`, non-greedy

## Phase 1: DGEN Grammar Prototype

Design the `.dgen` file syntax for defining dialects. Write `regex.dgen` as the first real dialect definition. This is the core design work — the grammar must be general enough for future dialects but simple enough to parse.

### 1a. Design conceptual modeling for the dialect.

Decide what the types and ops are conceptually, along with anything else. Get solid on these so we're not co-desigining the dialect with the DGEN file syntax.

### 1b. Try several different dialect file syntaxes. Iterate towards something that feels good.

**Deliverable**: `regex/regex.dgen` populated with the regex dialect definition.

## Phase 2: Regex Parser

Parse regex strings directly into IR in the regex dialect (no intermediate AST).

- Lexer: tokenize regex string into literals, metacharacters (`*`, `+`, `?`, `|`, `(`, `)`, `.`)
- Parser: emit regex dialect ops directly (e.g. `regex.literal`, `regex.concat`, `regex.alt`, `regex.star`, `regex.plus`, `regex.optional`)

### 2a. Define the first parts of the DGEN data model in Mojo as structs.
### 2b. Hand-write "generated" Mojo code which DGEN should generate from `regex.dgen`, and test it.
### 2c. Write a simple parser from regex strings -> regex dialect types in Mojo.
### 2d. Implement reading/writing the DGEN IR ops for the manual regex dialect types.
### 2e. Test the parser with tests regex string -> regex IR.
### 2f. Build out enough of the DGEN code generation code to generate the regex target code.

**Deliverable**: Parser that turns regex strings into regex dialect IR. Test with IR printing/round-trip.

## Phase 3: NFA Dialect & Lowering Pass

Define a second dialect for NFAs. Create a pass that lowers regex IR → NFA IR.

Strawman implementation:
- Define NFA dialect in `nfa.dgen` (states, transitions, epsilon transitions, accept states)
- Implement Thompson's construction as a regex → NFA lowering pass
- Each regex op maps to an NFA fragment (start state, accept state)

**Deliverable**: NFA dialect definition. Lowering pass from regex → NFA. Test by dumping NFA IR.

## Phase 4: NFA Interpreter

Write an interpreter for the NFA dialect.

Strawman implementation:
- Thompson simulation: track set of current states
- For each input character, compute next state set via NFA transitions
- Match succeeds if any state in final set is an accept state
- O(n*m) where n = string length, m = NFA size — no backtracking

**Deliverable**: Working `match(pattern, string) -> bool`. First end-to-end regex execution.

## Phase 5: JIT Compilation

Compile regex → machine code by lowering to 

Strawman implementation:
- NFA → LLVM IR
- Each NFA state becomes a labeled block
- Character comparisons become conditional jumps
- State set tracking becomes register/stack operations
- Goal: compile fast (simpler than PCRE-JIT's optimizer), run faster than interpreter

JIT Backend options
- LLVM
  - Highest code quality
  - only backend supporting GPU codegen
  - Sloooow
- B3
- MIR
- Cranelift

MIR and Cranelift are both interesting. MIR is fast and very thing as a dependency. Cranelift has a number of folks claiming they like the JIT API, but it's in Rust.

Ultimately LLVM is probably the right call here, at least today. DGEN shouldn't lock in a backend, but LLVM is the only backend that supports GPUs and is also just changing fewer variables at once.

**Deliverable**: JIT-compiled regex matching passes same test as the interpreter

## Verification

Each phase has its own testable deliverable. End-to-end verification:
1. `regex.dgen` parses and defines valid dialect
2. Regex strings parse to correct ASTs (test against known patterns)
3. NFA construction produces correct state machines (test small patterns by hand)
4. Interpreter matches correctly (test suite of pattern/string/expected-result triples)
5. JIT matches produce identical results to interpreter (differential testing)
