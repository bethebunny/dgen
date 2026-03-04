; Keywords
["from" "import" "trait" "type" "op" "block"] @keyword

; Arrow operator
"->" @operator

; Assignment
"=" @operator

; Delimiters
[":" ","] @punctuation.delimiter

; Brackets
["(" ")"] @punctuation.bracket
["<" ">"] @punctuation.bracket

; Comments
(comment) @comment

; Declaration names
(trait_declaration name: (identifier) @type)
(type_declaration name: (identifier) @type)
(op_declaration name: (identifier) @function)

; Import module
(import_declaration module: (identifier) @module)

; Imported names (types/traits)
(import_declaration name: (identifier) @type)

; Data field names
(data_field name: (identifier) @property)

; Block declaration names
(block_declaration name: (identifier) @variable)

; Parameter and operand names
(parameter name: (identifier) @variable.parameter)
(operand name: (identifier) @variable.parameter)

; Default values
(parameter default: (identifier) @constant)
(operand default: (identifier) @constant)

; Type references in params/operands/fields/return types
(parameter type: (identifier) @type)
(operand type: (identifier) @type)
(data_field type: (identifier) @type)
(op_declaration return_type: (identifier) @type)

; Generic type name and arguments
(generic_type name: (identifier) @type)
(generic_type argument: (identifier) @type)
