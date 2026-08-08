use bevy::prelude::*;

pub struct CounterPlugin;

impl Plugin for CounterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlobalIDCounter>();
    }
}

/// Allocates the handles clients use to refer to objects.
///
/// One sequence for everything in the world, so a handle is unambiguous
/// whatever it names.
#[derive(Resource, Default)]
pub struct GlobalIDCounter(u64);

impl GlobalIDCounter {
    /// Returns the current ID and increments the internal counter
    pub fn next(&mut self) -> u64 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

/// The handle an object is known by. Assigned at creation from
/// [`GlobalIDCounter`], and never reused.
#[derive(Component)]
pub struct UniqueID(pub u64);
