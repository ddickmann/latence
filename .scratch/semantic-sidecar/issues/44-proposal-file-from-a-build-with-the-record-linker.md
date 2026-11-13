# Task: the proposal file from a fresh build with the record linker

Type: task
Mode: HITL (it needs the machine that holds the linker's environment)
Status: open, with the owner's run of the handoff's block 8
Blocked by: nothing
Emerges from: 40 ("Not done")

## Question

The trial of 12 Nov wrote the proposal file twice: from the export kept on 11 Nov, which holds
the machine's 938 links, and from a fresh build without the record linker, because the
linker's environment was not on the machine that built the writer. Both files are the same
bytes, since no link is ever written into a proposal. A fresh build with the linker has not
been put through the command.

## Goal

One fresh build of the vehicle estate with the record linker on the development machine, the
proposal file written from it, compared byte for byte with the file of 12 Nov.

## Done check

- The comparison's result is written here.
- If the bytes differ, the entry that differs is named and has a ticket of its own.
