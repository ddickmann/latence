# Build: classes from an outside vocabulary as a third voice

Type: task
Mode: AFK
Status: open
Blocked by: nothing
Emerges from: 42 (split out on 14 Nov: the third of its four parts)

## Question

The class tree has two voices: attribute patterns from the catalog, and a language model. Ticket
34 names an outside vocabulary (a published class hierarchy) as a third voice. Which vocabulary,
under which licence, and how does it vote?

## Goal

A voice that reads a vocabulary file (classes, labels, parents) given in the estate file, proposes
a parent class for a concept where a label or synonym of the concept matches a class of the
vocabulary, and votes like the other two (a disagreement is contested, never settled by the
voice alone). The vocabulary's licence is checked and recorded before it is used.

## Done check

- Tests on the toy estate with a small vocabulary file: a match proposes, a disagreement with
  the other voices is contested, no match proposes nothing.
- The vocabulary for the vehicle estate is chosen with its licence recorded.
