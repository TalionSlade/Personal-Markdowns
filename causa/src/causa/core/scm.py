"""Structural Causal Model (Pearl 2009, §1.4).

A Structural Causal Model is a tuple :math:`(U, V, F, P(U))` where:

- :math:`U` is a set of exogenous (unobserved) noise variables;
- :math:`V` is a set of endogenous (observed) variables;
- :math:`F = \\{f_i\\}` is a set of structural equations
  :math:`V_i = f_i(\\text{parents}(V_i), U_i)`;
- :math:`P(U)` is the joint distribution over the noise variables.

This module is **declarative** — it captures the *shape* of the SCM (variables,
typed roles, structural equations).  The numerical estimation of P(V | do(·))
happens in :mod:`causa.planning.dowhy_scorer` against DoWhy.  We separate the
two so that the SCM specification stays auditable independently of the
estimator chosen for any given run (which is exactly what the dissertation's
ablation §J3 requires).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from causa.core.graph import CausalEdge, CausalGraph


class VariableType(str, Enum):
    """Typed role of a variable in the agent's perception loop.

    The distinction matters because different types feed different parts of
    the planning layer:

    - ``OBSERVATIONAL`` — sensed inputs; never intervened on
    - ``HYPOTHESIS``    — latent belief states (e.g. ``hypothesis_space``)
    - ``ACTION``        — the do-variable; this is where the agent intervenes
    - ``OUTCOME``       — observable success signals (e.g. ``tests_passed``)
    """

    OBSERVATIONAL = "observational"
    HYPOTHESIS = "hypothesis"
    ACTION = "action"
    OUTCOME = "outcome"


class VariableDomain(str, Enum):
    """Statistical domain of a variable's support."""

    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass(frozen=True)
class Variable:
    """A single SCM variable, fully specified."""

    name: str
    role: VariableType
    domain: VariableDomain
    description: str = ""
    levels: tuple[str, ...] = ()
    """For categorical/ordinal vars: the allowed level names, in order."""

    def __post_init__(self) -> None:
        if self.domain in (VariableDomain.CATEGORICAL, VariableDomain.ORDINAL):
            if not self.levels:
                raise ValueError(
                    f"Variable {self.name}: {self.domain.value} requires non-empty levels"
                )
        if self.domain == VariableDomain.BINARY and self.levels and len(self.levels) != 2:
            raise ValueError(f"Variable {self.name}: binary requires exactly 2 levels")


@dataclass(frozen=True)
class StructuralEquation:
    """A single structural equation V_i = f_i(parents, noise).

    The callable ``f`` is **not required** for the planning layer — DoWhy
    operates from the graph plus observational data.  We keep it optional so
    the SCM can be *simulated* in tests and for the synthetic warm-start
    generator (:mod:`causa.planning.warm_start`).
    """

    target: str
    parents: tuple[str, ...]
    f: Callable[[Mapping[str, Any], float], Any] | None = None
    noise_scale: float = 1.0

    def evaluate(self, parent_values: Mapping[str, Any], noise: float) -> Any:
        if self.f is None:
            raise NotImplementedError(
                f"Structural equation for {self.target!r} not provided; "
                "simulation requires explicit f."
            )
        return self.f(parent_values, noise)


@dataclass
class SCM:
    """A Structural Causal Model.

    Attributes
    ----------
    variables:
        Map from name to :class:`Variable`.  Order is insertion order.
    graph:
        The DAG over endogenous variables.  Edges describe direct causal
        influence; consistency with ``variables`` is enforced.
    equations:
        Map from target name to :class:`StructuralEquation`.  Optional —
        present only for SCMs we can simulate.
    metadata:
        Free-form authoring metadata (provenance, version, paper section).
    """

    variables: dict[str, Variable] = field(default_factory=dict)
    graph: CausalGraph = field(default_factory=CausalGraph)
    equations: dict[str, StructuralEquation] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────

    def add_variable(self, var: Variable) -> None:
        if var.name in self.variables:
            raise ValueError(f"variable {var.name!r} already present")
        self.variables[var.name] = var
        self.graph.add_node(var.name)

    def add_edge(self, source: str, target: str, *, semantics: str = "") -> None:
        if source not in self.variables:
            raise KeyError(f"unknown source variable: {source!r}")
        if target not in self.variables:
            raise KeyError(f"unknown target variable: {target!r}")
        self.graph.add_edge(CausalEdge(source, target, semantics=semantics))

    def set_equation(self, eq: StructuralEquation) -> None:
        if eq.target not in self.variables:
            raise KeyError(f"unknown target variable: {eq.target!r}")
        if set(eq.parents) != self.graph.parents(eq.target):
            raise ValueError(
                f"equation parents {eq.parents} for {eq.target!r} disagree "
                f"with graph parents {self.graph.parents(eq.target)}"
            )
        self.equations[eq.target] = eq

    # ── accessors ───────────────────────────────────────────────────────────

    def variables_by_role(self, role: VariableType) -> tuple[Variable, ...]:
        return tuple(v for v in self.variables.values() if v.role == role)

    @property
    def action_variable(self) -> Variable:
        """The unique do-variable for this SCM.

        Raises if zero or more than one variable is tagged
        :attr:`VariableType.ACTION` — the agent's planning loop assumes a
        single intervention point.
        """
        actions = self.variables_by_role(VariableType.ACTION)
        if len(actions) != 1:
            raise ValueError(
                f"expected exactly 1 ACTION variable, found {len(actions)}"
            )
        return actions[0]

    @property
    def outcome_variable(self) -> Variable:
        """The unique outcome variable (the success signal)."""
        outcomes = self.variables_by_role(VariableType.OUTCOME)
        if len(outcomes) != 1:
            raise ValueError(
                f"expected exactly 1 OUTCOME variable, found {len(outcomes)}"
            )
        return outcomes[0]

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise on structural inconsistency.

        Checks:
        1. The graph node set equals the variable name set.
        2. Exactly one ACTION variable; exactly one OUTCOME variable.
        3. Every variable with parents and an equation has matching parent sets.
        """
        node_set = set(self.graph.nodes)
        var_set = set(self.variables)
        if node_set != var_set:
            extra = node_set - var_set
            missing = var_set - node_set
            raise ValueError(
                f"SCM/graph mismatch: extra graph nodes={extra}, missing={missing}"
            )
        _ = self.action_variable  # raises if not exactly 1
        _ = self.outcome_variable  # raises if not exactly 1
        for tgt, eq in self.equations.items():
            if set(eq.parents) != self.graph.parents(tgt):
                raise ValueError(
                    f"equation parents for {tgt!r} disagree with graph"
                )

    def is_markovian(self) -> bool:
        """An SCM is *Markovian* iff noise terms are mutually independent.

        We treat the SCM as Markovian iff every endogenous variable has a
        defined equation with independent noise — i.e. no explicit
        unobserved confounders.  Non-Markovian SCMs (with bidirected edges)
        are out of scope for the M.Tech timeline (see dissertation §A8).
        """
        return all(v.name in self.equations or v.role == VariableType.OBSERVATIONAL
                   for v in self.variables.values())

    # ── pretty printing ─────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"SCM: {self.metadata.get('name', 'unnamed')}",
            f"  variables: {len(self.variables)} "
            f"({sum(1 for v in self.variables.values() if v.role == VariableType.OBSERVATIONAL)} obs, "
            f"{sum(1 for v in self.variables.values() if v.role == VariableType.HYPOTHESIS)} hyp, "
            f"{sum(1 for v in self.variables.values() if v.role == VariableType.ACTION)} act, "
            f"{sum(1 for v in self.variables.values() if v.role == VariableType.OUTCOME)} out)",
            f"  edges: {len(self.graph.edges)}",
            f"  equations: {len(self.equations)}",
            f"  markovian: {self.is_markovian()}",
        ]
        return "\n".join(lines)


class SCMBuilder:
    """Builder for :class:`SCM` instances — used by the domain module to
    declare the 9-node debugging SCM cleanly.

    Example
    -------
    >>> b = SCMBuilder("debugging_scm_v1")
    >>> b.observational("error_message_type", VariableDomain.CATEGORICAL,
    ...                 levels=("type_error", "value_error", "ref_error"))
    >>> b.action("tool_selected", VariableDomain.CATEGORICAL,
    ...          levels=("code_search", "test_runner"))
    >>> b.outcome("tests_passed", VariableDomain.BINARY,
    ...           levels=("pass", "fail"))
    >>> b.edge("error_message_type", "tool_selected", "error kind shapes tool choice")
    >>> scm = b.build()
    """

    def __init__(self, name: str) -> None:
        self._scm = SCM(metadata={"name": name})

    def _add(self, name: str, role: VariableType, domain: VariableDomain,
             description: str, levels: Iterable[str]) -> SCMBuilder:
        self._scm.add_variable(Variable(
            name=name, role=role, domain=domain,
            description=description, levels=tuple(levels),
        ))
        return self

    def observational(self, name: str, domain: VariableDomain, *,
                      description: str = "", levels: Iterable[str] = ()) -> SCMBuilder:
        return self._add(name, VariableType.OBSERVATIONAL, domain, description, levels)

    def hypothesis(self, name: str, domain: VariableDomain, *,
                   description: str = "", levels: Iterable[str] = ()) -> SCMBuilder:
        return self._add(name, VariableType.HYPOTHESIS, domain, description, levels)

    def action(self, name: str, domain: VariableDomain, *,
               description: str = "", levels: Iterable[str] = ()) -> SCMBuilder:
        return self._add(name, VariableType.ACTION, domain, description, levels)

    def outcome(self, name: str, domain: VariableDomain, *,
                description: str = "", levels: Iterable[str] = ()) -> SCMBuilder:
        return self._add(name, VariableType.OUTCOME, domain, description, levels)

    def edge(self, source: str, target: str, semantics: str = "") -> SCMBuilder:
        self._scm.add_edge(source, target, semantics=semantics)
        return self

    def build(self) -> SCM:
        self._scm.validate()
        return self._scm
