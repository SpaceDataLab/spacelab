//! Position and PV as measurements: fitting an orbit to a state vector somebody else published.
//!
//! Every other observable in this crate is a *projection* of the state — a distance, an angle, a
//! rate — and loses information on the way. An ephemeris product does not: a CCSDS OEM record, a
//! precise orbit solution, a GNSS navigation fix is the state itself, in some frame, at some epoch.
//! Fitting to it is therefore the trivial observable, and the reason to have it at all is that once
//! it is one it goes through the *same* machinery as every other: the same [`Observable`] trait, the
//! same automatic-differentiation partials, the same modifier chain, the same weighted least
//! squares. A special-cased "fit to states" path would be a second estimator to keep in step.
//!
//! # Why a component and not a vector
//!
//! [`Observable`] is scalar, and so is the filter's intake:
//! [`ResidualModel`](crate::gauss_newton::ResidualModel) emits one Jacobian *row* at a time, and
//! [`Srif`](crate::srif::Srif) folds one row at a time. A position measurement is three rows and a
//! PV measurement is six; there is nothing in between for a vector type to buy, and a vector
//! observable would have to be unpacked into rows at every call site anyway. So the unit here is one
//! component, [`StateComponent`], and [`Component::POSITION`] and [`Component::ALL`] are the two
//! groupings that have names.
//!
//! This is also what makes the correlations honest. Three separately weighted rows are exactly a
//! diagonal position covariance; an ephemeris that publishes a *full* covariance is not diagonal,
//! and the right answer there is to rotate the residual into the frame that diagonalises it before
//! it is emitted — which the row form permits and a fused vector observable would hide.
//!
//! # Frames and epochs are the caller's
//!
//! Nothing here knows what frame the state is in. The published state and the propagated one must
//! be in the same frame at the same epoch, and an OEM in ITRF fitted against a GCRF propagation is
//! a rotation error of thousands of kilometres that this layer cannot see. `spacelab_ndm` reads the
//! OEM; `spacelab_frames` rotates it; this evaluates the residual.

use spacelab_autodiff::Scalar;

use crate::modifier::Observable;

/// One component of the six-element state.
///
/// An enumeration rather than a checked index: there are exactly six, they are known at compile
/// time, and an out-of-range component then cannot be constructed rather than having to be rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    /// Position along the first axis.
    X,
    /// Position along the second axis.
    Y,
    /// Position along the third axis.
    Z,
    /// Velocity along the first axis.
    Vx,
    /// Velocity along the second axis.
    Vy,
    /// Velocity along the third axis.
    Vz,
}

impl Component {
    /// The three position components — a CCSDS OEM record without velocity, or any position-only
    /// product.
    pub const POSITION: [Self; 3] = [Self::X, Self::Y, Self::Z];
    /// The three velocity components.
    pub const VELOCITY: [Self; 3] = [Self::Vx, Self::Vy, Self::Vz];
    /// All six — the PV measurement.
    pub const ALL: [Self; 6] = [Self::X, Self::Y, Self::Z, Self::Vx, Self::Vy, Self::Vz];

    /// Where this component sits in a `[x, y, z, vx, vy, vz]` state.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::Vx => 3,
            Self::Vy => 4,
            Self::Vz => 5,
        }
    }

    /// Whether this component is a velocity — the two halves carry different units, and a caller
    /// weighting a PV measurement needs to know which sigma to apply.
    #[must_use]
    pub const fn is_velocity(self) -> bool {
        self.index() >= 3
    }
}

/// One component of the state, read directly — the [`Observable`] a published ephemeris is fitted
/// through.
///
/// Its partial with respect to the state is the unit row `eᵢ`, exactly, which is the whole content
/// of the model: an ephemeris measurement adds information to one state direction at the epoch it
/// is given, and the state transition matrix carries it back to the fit epoch like any other
/// observation's partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateComponent {
    /// The component measured.
    pub component: Component,
}

impl StateComponent {
    /// The observable that reads `component`.
    #[must_use]
    pub const fn new(component: Component) -> Self {
        Self { component }
    }

    /// The three position observables, in axis order.
    #[must_use]
    pub const fn position() -> [Self; 3] {
        [
            Self::new(Component::X),
            Self::new(Component::Y),
            Self::new(Component::Z),
        ]
    }

    /// All six observables of a PV measurement, in `[x, y, z, vx, vy, vz]` order.
    #[must_use]
    pub const fn position_velocity() -> [Self; 6] {
        [
            Self::new(Component::X),
            Self::new(Component::Y),
            Self::new(Component::Z),
            Self::new(Component::Vx),
            Self::new(Component::Vy),
            Self::new(Component::Vz),
        ]
    }
}

impl<const K: usize> Observable<K> for StateComponent {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        satellite[self.component.index()]
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::needless_range_loop)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;
    use crate::modifier::{Bias, EstimatedBias, Modified, partials};

    const STATE: [f64; 6] = [
        3_961_000.0,
        2_614_000.0,
        5_215_000.0,
        -3_140.0,
        6_190.0,
        -700.0,
    ];

    /// Each component reads the state slot it names, to the bit, and the six of a PV measurement
    /// reproduce the whole state.
    #[test]
    fn every_component_reads_the_state_slot_it_names() {
        let none: [f64; 0] = [];
        for (i, observable) in StateComponent::position_velocity().into_iter().enumerate() {
            assert_eq!(observable.component.index(), i);
            assert_within_ulps(observable.value(&STATE, &none), STATE[i], 0, "a component");
        }
        for (i, component) in Component::ALL.into_iter().enumerate() {
            assert_eq!(component.index(), i);
            assert_eq!(component.is_velocity(), i >= 3);
        }
        assert_eq!(
            Component::POSITION,
            [Component::X, Component::Y, Component::Z]
        );
        assert_eq!(
            Component::VELOCITY,
            [Component::Vx, Component::Vy, Component::Vz]
        );
        assert_eq!(
            StateComponent::position(),
            [
                StateComponent::new(Component::X),
                StateComponent::new(Component::Y),
                StateComponent::new(Component::Z)
            ]
        );
    }

    /// The partial is the unit row and nothing else — every other state direction exactly zero, not
    /// merely small, because the observable is a read rather than an arithmetic expression.
    #[test]
    fn the_partial_is_the_unit_row() {
        let none: [f64; 0] = [];
        for (i, observable) in StateComponent::position_velocity().into_iter().enumerate() {
            let p = partials(&observable, &STATE, &none);
            assert_within_ulps(p.value, STATE[i], 0, "the value");
            for j in 0..6 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_within_ulps(p.state[j], want, 0, "a partial");
            }
        }
    }

    /// A component observable is an [`Observable`] like any other, so the modifier chain applies to
    /// it: a published ephemeris carrying a known frame offset takes a [`Bias`], and one whose
    /// offset is unknown takes an [`EstimatedBias`] with its unit column.
    #[test]
    fn a_component_takes_the_modifier_chain() {
        let observable = Modified::new(StateComponent::new(Component::Y), Bias { value: -12.5 })
            .with(EstimatedBias::<1>::new(0).expect("the only parameter"));
        let p = partials(&observable, &STATE, &[4.25]);
        assert_within_ulps(
            p.value,
            STATE[1] - 12.5 + 4.25,
            0,
            "the corrected component",
        );
        assert_within_ulps(p.parameters[0], 1.0, 0, "∂component/∂bias");
        assert_within_ulps(p.state[1], 1.0, 0, "the unit partial survives the chain");
    }
}
