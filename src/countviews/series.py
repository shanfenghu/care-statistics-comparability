"""Compile aggregate rewrites over bags of finite event histories.

Each input machine assigns an integer contribution to one person's history;
its aggregate is the sum over the population. Rational weighted-automaton
forward-space saturation is classical (Schuetzenberger, Tzeng and Kiefer).
This module exposes that machinery for simultaneous aggregate reconstruction,
including constructive two-population obstructions. It does not infer programs
from prose, solve arbitrary SQL, or validate recorded person identities.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence, Tuple

from .linear import Obstruction, Rewrite, derive

Word = Tuple[str, ...]


@dataclass(frozen=True)
class Machine:
    """An integer linear representation alpha*M(word)*eta.

    Transition rows are sparse sequences of (destination, integer weight).
    Alphabet order is fixed across all machines in a compilation. Terminal
    values may be signed for contrasts; primitive counting programs are
    nonnegative. No assumption about nonnegative weights is needed by the
    equivalence algorithm, only integer contributions by the inputs.
    """
    alphabet: Tuple[str, ...]
    initial: Tuple[int, ...]
    transitions: Mapping[str, Tuple[Tuple[Tuple[int, int], ...], ...]]
    terminal: Tuple[int, ...]

    def __post_init__(self):
        n=len(self.initial)
        if not n or len(self.terminal)!=n or len(set(self.alphabet))!=len(self.alphabet):
            raise ValueError('Nonempty dimension, matching terminal and distinct symbols required')
        if set(self.transitions)!=set(self.alphabet):raise ValueError('Transition alphabet mismatch')
        if any(len(t)!=n for t in self.transitions.values()):raise ValueError('Transition shape mismatch')
        values=list(self.initial)+list(self.terminal)
        for matrix in self.transitions.values():
            for row in matrix:
                for target,weight in row:
                    if not isinstance(target,int) or isinstance(target,bool):
                        raise TypeError('Transition destinations must be integer state indices')
                    if not 0<=target<n:raise ValueError('Transition destination out of range')
                    values.append(weight)
        if any(not isinstance(v,int) or isinstance(v,bool) for v in values):
            raise TypeError('Machines require integer representation coefficients')

    @property
    def dimension(self):return len(self.initial)

    def step(self, vector, symbol):
        """Apply one sparse transition, preserving exact arithmetic."""
        result=[0]*self.dimension
        for value,row in zip(vector,self.transitions[symbol]):
            if value:
                for target,weight in row:result[target]+=value*weight
        return tuple(result)

    def evaluate(self, word: Sequence[str]):
        """Contribution of one declared history, with no real-data access."""
        vector=self.initial
        for symbol in word:vector=self.step(vector,symbol)
        return sum(v*t for v,t in zip(vector,self.terminal))


def indicator(alphabet, transitions, initial, accepting):
    """Represent a complete deterministic finite-state predicate as 0/1.

    State indices must be Python integers, excluding booleans. In particular,
    fractional initial/accepting states must not silently become an all-zero
    predicate through equality comparisons with the integer state numbers.
    """
    alphabet=tuple(alphabet);accepting=tuple(accepting)
    n=len(transitions)
    indices=[initial,*accepting,*[v for row in transitions for v in row]]
    if any(not isinstance(v,int) or isinstance(v,bool) for v in indices):
        raise TypeError('State indices must be integers, excluding booleans')
    if not n or any(len(row)!=len(alphabet) for row in transitions):
        raise ValueError('A nonempty complete transition table is required')
    if any(not 0<=v<n for v in indices):
        raise ValueError('State index out of range')
    start=tuple(int(i==initial) for i in range(n))
    final=tuple(int(i in accepting) for i in range(n))
    matrices={a:tuple(((row[j],1),) for row in transitions) for j,a in enumerate(alphabet)}
    return Machine(tuple(alphabet),start,matrices,final)


def counter(alphabet, increments):
    """Count symbol occurrences, allowing integer weights for contrasts."""
    matrices={a:(((0,1),(1,increments.get(a,0))),((1,1),)) for a in alphabet}
    return Machine(tuple(alphabet),(1,0),matrices,(0,1))


def _joint_system(machines):
    """Share literally identical initial/transition structures across outputs."""
    if not machines:raise ValueError('At least one machine required')
    alphabet=machines[0].alphabet
    initial=[];matrices={a:[] for a in alphabet};groups={};offsets=[]
    for m in machines:
        if m.alphabet!=alphabet:raise ValueError('Alphabets differ')
        key=(m.initial,tuple(m.transitions[a] for a in alphabet))
        if key not in groups:
            offset=len(initial);groups[key]=offset;initial.extend(m.initial)
            for a in alphabet:matrices[a].extend(tuple((j+offset,v) for j,v in row) for row in m.transitions[a])
        offsets.append(groups[key])
    return Machine(alphabet,tuple(initial),{a:tuple(rows) for a,rows in matrices.items()},tuple(0 for _ in initial)),offsets


def combination(machines, weights):
    """Direct-sum representation of a rational linear combination.

    Zero coefficients are omitted. Rational weights are kept outside the
    integer Machine constructor by multiplying the final evaluation vector.
    The returned pair is a joint initial/transition representation and an
    exact output vector, so the expensive state space depends on used views.
    """
    selected=[(m,Fraction(w)) for m,w in zip(machines,weights) if w]
    if len(machines)!=len(weights):raise ValueError('One weight per machine required')
    if not selected:
        return None,()
    joint,offsets=_joint_system([m for m,_ in selected]);terminal=[Fraction(0)]*joint.dimension
    for (m,w),offset in zip(selected,offsets):
        for i,value in enumerate(m.terminal):terminal[offset+i]+=w*value
    return joint,tuple(terminal)


def restrict(machine, domain):
    """Mask a series by a deterministic 0/1 legal-history machine.

    The caller supplies a domain made by indicator(). This construction is
    the classical Hadamard product. It multiplies dimensions by domain size,
    rather than multiplying all view-machine state counts together.
    """
    if machine.alphabet!=domain.alphabet:raise ValueError('Alphabets differ')
    if (sum(domain.initial)!=1 or any(v not in (0,1) for v in domain.initial+domain.terminal)
            or any(len(row)!=1 or row[0][1]!=1 for matrix in domain.transitions.values() for row in matrix)):
        raise ValueError('Legal-history domain must be a complete deterministic 0/1 automaton')
    d=domain.dimension;n=machine.dimension
    initial=tuple(x*y for x in machine.initial for y in domain.initial)
    final=tuple(x*y for x in machine.terminal for y in domain.terminal)
    matrices={}
    for a in machine.alphabet:
        rows=[]
        for row in machine.transitions[a]:
            for drow in domain.transitions[a]:
                rows.append(tuple((j*d+k,w*v) for j,w in row for k,v in drow))
        matrices[a]=tuple(rows)
    return Machine(machine.alphabet,initial,matrices,final)


class Echelon:
    """Incremental exact independence test; unnormalised originals stay separate."""
    def __init__(self):self.rows=[]

    def add(self, vector):
        row=list(map(Fraction,vector))
        for pivot,basis in self.rows:
            if row[pivot]:
                scale=row[pivot]
                row=[x-scale*y for x,y in zip(row,basis)]
        pivot=next((i for i,v in enumerate(row) if v),None)
        if pivot is None:return False
        scale=row[pivot];row=[v/scale for v in row]
        self.rows.append((pivot,row))
        # Pivot order can differ from insertion order; future elimination must
        # respect increasing pivots. Older rows need not be reduced above it.
        self.rows.sort(key=lambda pair:pair[0])
        return True


def forward_basis(machine, output=None, outputs=None):
    """Classical exact forward saturation; optionally stop on a nonzero output.

    Returns (basis word/vector pairs, separating word or None, steps).
    When output is supplied and no separator exists, the returned span is
    closed under every transition and orthogonal to that output vector.
    Dependent original rows are not expanded: their successors are linear
    combinations of successors of the retained basis rows.
    """
    if output is not None and outputs is not None:raise ValueError('Choose scalar or vector output')
    tests=outputs if outputs is not None else ([output] if output is not None else None)
    basis=Echelon();queue=deque();pairs=[];steps=0
    if basis.add(machine.initial):
        queue.append(((),machine.initial));pairs.append(((),machine.initial))
    while queue:
        word,vector=queue.popleft()
        if tests is not None and any(sum(v*t for v,t in zip(vector,test)) for test in tests):
            return pairs,word,steps
        for symbol in machine.alphabet:
            next_vector=machine.step(vector,symbol);steps+=1
            if basis.add(next_vector):
                word2=word+(symbol,)
                pairs.append((word2,next_vector));queue.append((word2,next_vector))
    return pairs,None,steps


@dataclass(frozen=True)
class PopulationWitness:
    """Two finite multisets of histories indistinguishable to all input views."""
    words: Tuple[Word, ...]
    left: Tuple[int, ...]
    right: Tuple[int, ...]


@dataclass(frozen=True)
class Compilation:
    """A rewrite or population witness, with explicit compilation work counters."""
    result: object
    samples: Tuple[Word, ...]
    oracle_calls: int
    saturated_rows: int
    transition_steps: int
    total_dimension: int


def _prepare(views,query,domain):
    machines=list(views)+[query]
    if any(m.alphabet!=query.alphabet for m in machines):raise ValueError('Alphabets differ')
    return [restrict(m,domain) if domain is not None else m for m in machines]


def _decision(machines,samples):
    profiles=[[m.evaluate(w) for w in samples] for m in machines]
    if not samples:return Rewrite(tuple(Fraction(0) for _ in machines[:-1]))
    result=derive(profiles[:-1],profiles[-1])
    if isinstance(result,Obstruction):
        result=PopulationWitness(tuple(samples),result.left,result.right)
    return result


def compile_direct(views,query,domain=None):
    """Complete aggregate determinacy by one classical joint-state saturation."""
    machines=_prepare(views,query,domain)
    joint,_=_joint_system(machines)
    pairs,_,steps=forward_basis(joint)
    samples=tuple(word for word,_ in pairs)
    return Compilation(_decision(machines,samples),samples,1,len(pairs),steps,joint.dimension)


def compile_batch_direct(views,queries,domain=None):
    """Strong batch baseline: one shared classical saturation for all targets."""
    if not queries:return ()
    machines=_prepare(views,queries[0],domain)[:-1]
    targets=[restrict(q,domain) if domain is not None else q for q in queries]
    joint,_=_joint_system(machines+targets)
    pairs,_,steps=forward_basis(joint);samples=tuple(w for w,_ in pairs)
    return tuple(Compilation(_decision(machines+[q],samples),samples,
                             int(i==0),len(pairs) if i==0 else 0,
                             steps if i==0 else 0,joint.dimension)
                 for i,q in enumerate(targets))


def compile_refine(views,query,domain=None,seeds=()):
    """Refine a candidate linear rewrite with counterexample histories.

    Each unsuccessful equivalence check adds a new constraint on the rewrite
    coefficients. With m views, at most m+1 such checks are needed (possibly
    ending in a population obstruction). Equivalence checking itself is the
    classical weighted-automaton routine, not a new primitive.
    """
    machines=_prepare(views,query,domain);samples=list(dict.fromkeys(tuple(w) for w in seeds))
    calls=rows=steps=0
    while True:
        result=_decision(machines,samples)
        if isinstance(result,PopulationWitness):break
        joint,output=combination(machines,list(result.coefficients)+[-1])
        calls+=1
        if joint is None:break
        pairs,word,work=forward_basis(joint,output)
        rows+=len(pairs);steps+=work
        if word is None:break
        if word in samples:raise AssertionError('Separator must violate a new coefficient constraint')
        samples.append(word)
        if calls>len(views)+1:raise AssertionError('Coefficient refinement bound exceeded')
    return Compilation(result,tuple(samples),calls,rows,steps,sum(m.dimension for m in machines))


def verify_witness(views,query,witness,domain=None):
    """Independently evaluate all positive-multiplicity histories in a witness."""
    if len(witness.words)!=len(witness.left) or len(witness.words)!=len(witness.right):return False
    if any(not isinstance(n,int) or isinstance(n,bool) or n<0
           for n in witness.left+witness.right):return False
    if domain is not None:
        if any((l or r) and domain.evaluate(w)!=1 for w,l,r in zip(witness.words,witness.left,witness.right)):return False
    def total(machine,counts):return sum(n*machine.evaluate(w) for n,w in zip(counts,witness.words))
    return (all(total(m,witness.left)==total(m,witness.right) for m in views)
            and total(query,witness.left)!=total(query,witness.right))


def compile_workload(views,queries,domain=None):
    """Share separating histories across a fixed-view query workload.

    Every newly added history is independent in the output-profile space of
    all views and queries. If that rank is r and k queries are requested,
    at most r separating histories and r+k equivalence-oracle calls occur.
    This bounds oracle *calls*, not their state-space cost or wall time.
    """
    samples=[];results=[]
    for query in queries:
        result=compile_refine(views,query,domain,seeds=samples)
        samples=list(result.samples);results.append(result)
    return tuple(results)


def compile_vector(views,queries,domain=None):
    """Refine all surviving targets with a single vector equivalence oracle.

    Targets already separated by the sample histories receive population
    witnesses and leave the oracle. Remaining candidate rewrites are checked
    simultaneously, sharing transition structures and omitting unused views.
    Each separator either raises view-profile rank or rejects a target. With
    m views and b negative targets, at most m+b+1 oracle calls are needed;
    the sharper profile-rank bound is r+1. These count calls, not arithmetic.
    """
    if not queries:return ()
    vm=[restrict(v,domain) if domain is not None else v for v in views]
    qm=[restrict(q,domain) if domain is not None else q for q in queries]
    samples=[];decisions={};calls=rows=steps=0;peak=0
    while True:
        pending={}
        for i,q in enumerate(qm):
            if i in decisions:continue
            decision=_decision(vm+[q],samples)
            if isinstance(decision,PopulationWitness):decisions[i]=decision
            else:pending[i]=decision
        if not pending:break
        used=[i for i in range(len(vm)) if any(r.coefficients[i] for r in pending.values())]
        machines=[vm[i] for i in used]+[qm[i] for i in pending]
        joint,offsets=_joint_system(machines);outputs=[]
        for j,(i,result) in enumerate(pending.items()):
            terminal=[Fraction(0)]*joint.dimension
            for machine,offset,w in zip(machines[:len(used)],offsets[:len(used)],
                                        [result.coefficients[v] for v in used]):
                for s,value in enumerate(machine.terminal):terminal[offset+s]+=w*value
            q=qm[i];offset=offsets[len(used)+j]
            for s,value in enumerate(q.terminal):terminal[offset+s]-=value
            outputs.append(tuple(terminal))
        pairs,word,work=forward_basis(joint,outputs=outputs)
        calls+=1;rows+=len(pairs);steps+=work;peak=max(peak,joint.dimension)
        if word is None:
            decisions.update(pending);break
        if word in samples:raise AssertionError('Vector separator repeats a satisfied sample')
        samples.append(word)
        if calls>len(views)+len(queries)+1:raise AssertionError('Vector refinement bound exceeded')
    return tuple(Compilation(decisions[i],tuple(samples),calls if i==0 else 0,
                             rows if i==0 else 0,steps if i==0 else 0,peak)
                 for i in range(len(queries)))
