package org.vadere.simulator.models.groups.sir;


import org.vadere.annotation.factories.models.ModelClass;
import org.vadere.simulator.models.Model;
import org.vadere.simulator.models.groups.AbstractGroupModel;
import org.vadere.simulator.models.groups.Group;
import org.vadere.simulator.models.groups.GroupSizeDeterminator;
import org.vadere.simulator.models.groups.cgm.CentroidGroup;
import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.simulator.projects.Domain;
import org.vadere.state.attributes.Attributes;
import org.vadere.simulator.models.groups.sir.SIRGroup;
import org.vadere.state.attributes.models.AttributesSIRG;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.scenario.DynamicElementContainer;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.Topography;
import org.vadere.util.geometry.LinkedCellsGrid;

import java.util.*;

/**
 * Implementation of groups for a susceptible / infected / removed (SIR) model.
 */
@ModelClass
public class SIRGroupModel extends AbstractGroupModel<SIRGroup> {

	private Random random;
	private LinkedHashMap<Integer, SIRGroup> groupsById;
	private Map<Integer, LinkedList<SIRGroup>> sourceNextGroups;
	private AttributesSIRG attributesSIRG;
	private Topography topography;
	private IPotentialFieldTarget potentialFieldTarget;
	private int totalInfected = 0;
	//counter which hold the current number of calls of update method
	private int callUpdate = 0;
	//initial value of the allowed number of updates to a specific simulation time
	private double permittedUpdate = 0;


	public SIRGroupModel() {
		this.groupsById = new LinkedHashMap<>();
		this.sourceNextGroups = new HashMap<>();
	}

	@Override
	public void initialize(List<Attributes> attributesList, Domain domain,
	                       AttributesAgent attributesPedestrian, Random random) {
		this.attributesSIRG = Model.findAttributes(attributesList, AttributesSIRG.class);
		this.topography = domain.getTopography();
		this.random = random;
        this.totalInfected = 0;
	}

	@Override
	public void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget) {
		this.potentialFieldTarget = potentialFieldTarget;
		// update all existing groups
		for (SIRGroup group : groupsById.values()) {
			group.setPotentialFieldTarget(potentialFieldTarget);
		}
	}

	@Override
	public IPotentialFieldTarget getPotentialFieldTarget() {
		return potentialFieldTarget;
	}

	private int getFreeGroupId() {
		if (this.random.nextDouble() < this.attributesSIRG.getInfectionRate()
        || this.totalInfected < this.attributesSIRG.getInfectionsAtStart()) {
			if(!getGroupsById().containsKey(SIRType.ID_INFECTED.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_INFECTED.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_INFECTED.ordinal(), g);
			}
            this.totalInfected += 1;
			return SIRType.ID_INFECTED.ordinal();
		}
		else{
			if(!getGroupsById().containsKey(SIRType.ID_SUSCEPTIBLE.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_SUSCEPTIBLE.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_SUSCEPTIBLE.ordinal(), g);
			}
			return SIRType.ID_SUSCEPTIBLE.ordinal();
		}
	}


	@Override
	public void registerGroupSizeDeterminator(int sourceId, GroupSizeDeterminator gsD) {
		sourceNextGroups.put(sourceId, new LinkedList<>());
	}

	@Override
	public int nextGroupForSource(int sourceId) {
		return Integer.MAX_VALUE/2;
	}

	@Override
	public SIRGroup getGroup(final Pedestrian pedestrian) {
		SIRGroup group = groupsById.get(pedestrian.getGroupIds().getFirst());
		assert group != null : "No group found for pedestrian";
		return group;
	}

	@Override
	protected void registerMember(final Pedestrian ped, final SIRGroup group) {
		groupsById.putIfAbsent(ped.getGroupIds().getFirst(), group);
	}

	@Override
	public Map<Integer, SIRGroup> getGroupsById() {
		return groupsById;
	}

	@Override
	protected SIRGroup getNewGroup(final int size) {
		return getNewGroup(getFreeGroupId(), size);
	}

	@Override
	protected SIRGroup getNewGroup(final int id, final int size) {
		if(groupsById.containsKey(id))
		{
			return groupsById.get(id);
		}
		else
		{
			return new SIRGroup(id, this);
		}
	}

	private void initializeGroupsOfInitialPedestrians() {
		// get all pedestrians already in topography
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();
		// initially add single pedestrians to groups which are not created by a source
		if (c.getElements().size() > 0) {
			for (Pedestrian pedestrian : c.getElements()) {
				this.assignToGroup(pedestrian);
			}
		}
	}

	protected void assignToGroup(Pedestrian ped, int groupId) {
		SIRGroup currentGroup = getNewGroup(groupId, Integer.MAX_VALUE/2);
		currentGroup.addMember(ped);
		ped.getGroupIds().clear();
		ped.getGroupSizes().clear();
		ped.addGroupId(currentGroup.getID(), currentGroup.getSize());
		registerMember(ped, currentGroup);
	}

	protected void assignToGroup(Pedestrian ped) {
		int groupId = getFreeGroupId();
		assignToGroup(ped, groupId);
	}


	/* DynamicElement Listeners */

	@Override
	public void elementAdded(Pedestrian pedestrian) {
		assignToGroup(pedestrian);
	}

	@Override
	public void elementRemoved(Pedestrian pedestrian) {
		Group group = groupsById.get(pedestrian.getGroupIds().getFirst());
		if (group.removeMember(pedestrian)) { // if true pedestrian was last member.
			groupsById.remove(group.getID());
		}
	}

	/* Model Interface */

	@Override
	public void preLoop(final double simTimeInSec) {
		initializeGroupsOfInitialPedestrians();
		topography.addElementAddedListener(Pedestrian.class, this);
		topography.addElementRemovedListener(Pedestrian.class, this);
	}

	@Override
	public void postLoop(final double simTimeInSec) {
	}

	/**
	 * new helper method which represents the actual update of the SIR pedestrians
	 */
	public void updateSIRpedestrians() {
		// check the positions of all pedestrians and switch groups to INFECTED (or REMOVED).
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();
		LinkedCellsGrid<Pedestrian> grid = topography.getSpatialMap(Pedestrian.class);

		if (c.getElements().size() > 0) {
			//loop over all pedestrians
			for (Pedestrian p : c.getElements()) {

				if (getGroup(p).getID() == SIRType.ID_REMOVED.ordinal()) {
					continue;
				} else if (getGroup(p).getID() == SIRType.ID_INFECTED.ordinal()) {
					if (this.random.nextDouble() < attributesSIRG.getRemoveProbability()) {
						//set to removed state
						elementRemoved(p);
						assignToGroup(p, SIRType.ID_REMOVED.ordinal());
						this.totalInfected -= 1;
					}
				} else {
					List<Pedestrian> p_neighbors = grid.getObjects(p.getPosition(), attributesSIRG.getInfectionMaxDistance());
					// loop over neighbors and set infected if we are close
					if (p_neighbors.size() > 0) {
						for (Pedestrian p_neighbor : p_neighbors) {
							if (p == p_neighbor || getGroup(p_neighbor).getID() != SIRType.ID_INFECTED.ordinal())
								continue;
							//infect pedestrian with probability of infection rate
							if (this.random.nextDouble() < attributesSIRG.getInfectionRate()) {
								if (getGroup(p).getID() == SIRType.ID_SUSCEPTIBLE.ordinal()) {
									elementRemoved(p);
									assignToGroup(p, SIRType.ID_INFECTED.ordinal());
									this.totalInfected += 1;
								}
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void update(final double simTimeInSec) {
		//if infection rate is independent of step length
		if (attributesSIRG.isControlUpdateFrequencyIndependentOfStepLength()) {
			//calculate the number of allowed updates
			permittedUpdate = simTimeInSec * attributesSIRG.getGoalFrequency();
			//call updateSIRpedestrians as often as allowed to be independent of step length
			while (callUpdate + 1 <= permittedUpdate) {
				//increase counter
				callUpdate += 1;
				//update SIR pedestrians
				updateSIRpedestrians();
			}
		} else {
			//if update dependent on step length call updateSIRpedestrians only once
			updateSIRpedestrians();
		}
	}
}