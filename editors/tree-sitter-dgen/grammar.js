/// <reference types="tree-sitter-cli/dsl" />

module.exports = grammar({
  name: "dgen",

  extras: $ => [/\s/, $.comment],

  word: $ => $.identifier,

  rules: {
    source_file: $ => repeat($._declaration),

    _declaration: $ => choice(
      $.import_declaration,
      $.trait_declaration,
      $.type_declaration,
      $.op_declaration,
    ),

    import_declaration: $ => seq(
      'from',
      field('module', $.identifier),
      'import',
      commaSep1(field('name', $.identifier)),
    ),

    trait_declaration: $ => seq(
      'trait',
      field('name', $.identifier),
    ),

    type_declaration: $ => seq(
      'type',
      field('name', $.identifier),
      optional(field('type_parameters', $.type_parameters)),
      optional(seq(':', repeat($.data_field))),
    ),

    data_field: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
    ),

    op_declaration: $ => seq(
      'op',
      field('name', $.identifier),
      optional(field('type_parameters', $.type_parameters)),
      '(',
      optional(field('operands', $.operand_list)),
      ')',
      '->',
      field('return_type', $._type_ref),
      optional(seq(':', repeat($.block_declaration))),
    ),

    block_declaration: $ => seq(
      'block',
      field('name', $.identifier),
    ),

    type_parameters: $ => seq(
      '<',
      commaSep1($.parameter),
      '>',
    ),

    parameter: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
      optional(seq('=', field('default', $.identifier))),
    ),

    operand_list: $ => commaSep1($.operand),

    operand: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type_ref),
      optional(seq('=', field('default', $.identifier))),
    ),

    _type_ref: $ => choice(
      $.generic_type,
      $.identifier,
    ),

    generic_type: $ => prec(1, seq(
      field('name', $.identifier),
      '<',
      commaSep1(field('argument', $._type_ref)),
      '>',
    )),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    comment: $ => token(seq('#', /.*/)),
  },
});

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
