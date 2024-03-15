# FlexIO

Flexible IO that supports reading data from multiple sources.

Contact: agritsenko@google.com dehghani@google.com mjlm@google.com

# Motivation
Deliver a flexible lightweight research-friendly input pipeline that enables quick hacking and experimentation, and can be used for many projects and tasks.

# Key requirements / features
 * Support common image and video dataset sources
 * Support extensible (per-dataset) pre-processing
 * Provide a clear overview of the pre-processing ops
 * Support fully deterministic pre-processing
 * Rely on a well-tested pre-processing op library
 * Support dataset mixing (e.g. for co-training on several datasets)
