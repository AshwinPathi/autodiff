extremely inefficient version of autodiff in C++. Only works on scalar values, unless you want to have an array of a bunch of shared pointers.

# TODO
- [ ] Clean up `node.h`
    - [ ] Optimize variable application
    - [ ] Cache forward passes
    - [ ] Make it less state-y
    - [ ] Possibly allow any arbitrary expression to `.get(variable)` to access its children variables with that name
    - [ ] Fix unary negation
    - [ ] Cleaner internal API to differentiate `Variable`/`Constant`
- [ ] Add Tensors
- [ ] GraphViz intgration
- [ ] Create Optimizer/Compiler
    - [ ] Dead code elim
    - [ ] Constant folding
    - [ ] Operator fusion
