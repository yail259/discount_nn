package main

import (
	"fmt"
	"math"
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
	nonlin bool
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

func Pow(a *Value, b *Value) Value {
	nv := Value{
		data:   math.Pow(a.data, b.data),
		grad:   0,
		op:     "pow",
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

	for i := 0; i < len(topo); i++ {
		if topo[i].op == "add" {
			topo[i].child1.grad += topo[i].grad
			topo[i].child2.grad += topo[i].grad
		} else if topo[i].op == "mult" {
			topo[i].child1.grad += topo[i].child2.data * topo[i].grad
			topo[i].child2.grad += topo[i].child1.data * topo[i].grad
		} else if topo[i].op == "pow" {
			topo[i].child1.grad += topo[i].child2.data * math.Pow(topo[i].child1.data, topo[i].child2.data-1) * topo[i].grad
		}
	}

	fmt.Println("TOPO:")
	for _, v := range topo {
		fmt.Printf("%+v\n", v)
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
		w[i] = NewValue(rand.NormFloat64() * 0.1)
	}

	neuron := Neuron{
		weight: w,
		bias:   NewValue(rand.NormFloat64() * 0.1),
		nonlin: true,
	}

	return neuron
}

func NewLayer(nin int, nout int) Layer {
	neurons := make([]Neuron, 0)

	for i := 0; i < nout; i++ {
		neurons = append(neurons, NewNeuron(nin))
	}

	layer := Layer{
		neurons: neurons,
	}

	return layer
}

func NewMLP(nin int, l_sizes []int) MLP {
	layers := make([]Layer, 0)

	layers = append(layers, NewLayer(nin, l_sizes[0]))

	for i := 1; i < len(l_sizes); i++ {
		layers = append(layers, NewLayer(l_sizes[i-1], l_sizes[i]))
	}

	for _, v := range layers[len(layers)-1].neurons {
		v.nonlin = false
	}

	mlp := MLP{
		layers: layers,
	}

	return mlp
}

func (n *Neuron) forward(x []Value) Value {
	total := &n.bias

	for i := 0; i < len(x); i++ {
		temp := Mult(&x[i], &n.weight[i])
		newTotal := Add(total, &temp)
		total = &newTotal
	}

	if n.nonlin && total.data < 0 {
		total.data = 0
	}

	return *total
}

func (l *Layer) forward(x []Value) []Value {
	res := make([]Value, 0)

	for _, v := range l.neurons {
		newX := v.forward(x)
		res = append(res, newX)
	}

	return res
}

func (mlp *MLP) forward(x []Value) []Value {
	inp := x

	for _, v := range mlp.layers {
		inp = v.forward(inp)
	}

	return inp
}

func Loss(actual []Value, target []Value) Value {
	loss := NewValue(0)
	lossp := &loss

	neg1 := NewValue(-1)
	pos2 := NewValue(2)

	for i := 0; i < len(actual); i++ {
		negTarget := Mult(&neg1, &target[i])
		diff := Add(&actual[i], &negTarget)

		squared := Pow(&diff, &pos2)
		newloss := Add(lossp, &squared)
		lossp = &newloss
	}

	return *lossp
}

func PrintMLP(mlp MLP) {
	for i, v := range mlp.layers {
		fmt.Printf("Layer %v\n", i)
		for _, n := range v.neurons {
			fmt.Println(n)
		}
	}
}

func main() {
	rand.Seed(69)

	// Basic calculation demo

	// a := NewValue(12.4)

	// x := Mult(&a, &a)
	// y := Add(&x, &a)
	// m := NewValue(2)
	// k := Pow(&y, &m)
	// n := Add(&a, &k)

	// Backward(&n)

	// ZeroGrad(&y)

	// fmt.Println(a)
	// fmt.Println(x)
	// fmt.Println(y)

	// fmt.Println(NewNeuron(10))

	mlp := NewMLP(2, []int{1})
	PrintMLP(mlp)

	x := []Value{NewValue(3), NewValue(4)}
	y := []Value{NewValue(2)}

	out := mlp.forward(x)

	loss := Loss(out, y)
	Backward(&loss)

	// fmt.Println(out)

	// topo := make([]*Value, 0)
	// visited := make(map[*Value]bool)

	// FindChildren(&out[0], &topo, &visited)
	// slices.Reverse(topo)

	// fmt.Println(topo)

	// tree.Print(&out[0])

	// PrintMLP(mlp)

	// fmt.Println(out, y)
}
