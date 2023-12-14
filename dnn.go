package main

import (
	"fmt"
	"math/rand"
	"slices"
)

type Value struct {
	data   float64
	grad   float64
	op     string
	child1 *Value
	child2 *Value
}

type Neuron struct {
	weight []Value
	bias   Value
}

type Layer struct {
	neurons []Neuron
}

type MLP struct {
	layers []Layer
}

func NewValue(data float64) Value {
	nv := Value{
		data:   data,
		grad:   0,
		op:     "",
		child1: nil,
		child2: nil,
	}

	return nv
}

func Mult(a *Value, b *Value) Value {
	nv := Value{
		data:   a.data * b.data,
		grad:   0,
		op:     "mult",
		child1: a,
		child2: b,
	}

	return nv
}

func Add(a *Value, b *Value) Value {
	nv := Value{
		data:   a.data + b.data,
		grad:   0,
		op:     "add",
		child1: a,
		child2: b,
	}

	return nv
}

func FindChildren(node *Value, topo *[]*Value, visited *map[*Value]bool) {
	if node != nil && !(*visited)[node] {
		(*visited)[node] = true

		FindChildren(node.child1, topo, visited)
		FindChildren(node.child2, topo, visited)

		*topo = append(*topo, node)
	}
}

func Backward(root *Value) {
	root.grad = 1

	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)

	FindChildren(root, &topo, &visited)
	slices.Reverse(topo)

	fmt.Println(topo)

	for _, v := range topo {
		if v.op == "add" {
			v.child1.grad += v.grad
			v.child2.grad += v.grad
		} else if v.op == "mult" {
			v.child1.grad += v.child2.data * v.grad
			v.child2.grad += v.child1.data * v.grad
		}
	}
}

func ZeroGrad(root *Value) {
	if root == nil {
		return
	}

	root.grad = 0

	ZeroGrad(root.child1)
	ZeroGrad(root.child2)
}

func NewNeuron(nin int) Neuron {
	w := make([]Value, nin)

	for i := 0; i < len(w); i++ {
		w[i] = NewValue(rand.NormFloat64())
	}

	neuron := Neuron{
		weight: w,
		bias:   NewValue(rand.NormFloat64()),
	}

	return neuron
}

func main() {
	a := NewValue(12.4)

	x := Mult(&a, &a)
	y := Add(&x, &a)

	Backward(&y)

	ZeroGrad(&y)

	fmt.Println(a)
	fmt.Println(x)
	fmt.Println(y)

	fmt.Println(NewNeuron(10))
}
