# Build: the neural second stage runs as a stage on the owner's machine

Type: task
Mode: HITL (a GPU: the owner's machine, until the deployment on AWS)
Status: open
Blocked by: a GPU
Emerges from: 13 (decided by the owner, 14 Nov: "the neural second stage (the small
re-scoring model and the reranker) is kept and runs on the owner's machine until the deployment
on AWS")

## Question

The spike measured the small re-scoring model (GLiDE 0.4B) and the reranker on the undecided
pairs: together they cut the review band by about two thirds at 10^5 records. They run only in
the spike's model arms. They are to become a stage of the estate build that runs where a GPU is,
and is skipped with a stated reason where none is.

## Goal

- A stage between the first-stage scorer and the review list, behind a seam (pairs in, scores
  out), with the model's name and version as the actor of every score it gives.
- Where no GPU is, the build says that the stage did not run and lists the pairs as before.

## Done check

- The review band on the vehicle estate with and without the stage, on the owner's machine.
- Offline tests with a faithful fake of the model's interface.

## Not done here

Nothing: the stage needs a GPU to be measured, and the brief of 14 Nov builds only what needs
none. The seam and its fake are the first step on the owner's machine.
