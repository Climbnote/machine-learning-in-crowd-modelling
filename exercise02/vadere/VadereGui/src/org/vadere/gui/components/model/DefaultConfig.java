package org.vadere.gui.components.model;

import org.vadere.state.psychology.cognition.GroupMembership;

import java.awt.*;
import java.util.HashMap;

public class DefaultConfig {

	// Member Variables
	private Color obstacleColor = new Color(0.7f,0.7f,0.7f);
	private Color sourceColor = new Color(0.3333333333333333f, 0.6588235294117647f, 0.40784313725490196f);
	private Color targetColor = new Color(0.8666666666666667f, 0.51764705882352946f, 0.32156862745098042f);
	private Color targetChangerColor = new Color(1.00f, 0.60f, 0.00f);
	private Color absorbingAreaColor = new Color(0.76863f,0.30588f, 0.32157f);
	private Color densityColor = Color.RED;
	private Color stairColor = new Color(0.5058823529411764f, 0.4470588235294118f, 0.6980392156862745f);
	private Color pedestrianColor = new Color(0.2980392156862745f, 0.4470588235294118f, 0.7901960784313725f);
	private Color measurementAreaColor = Color.RED;
	private int measurementAreaAlpha = 140;
	private HashMap<GroupMembership, Color> groupMembershipColors = new HashMap<>();
	private boolean changed = false;

	// Constructors
	public DefaultConfig() {
		initGroupMembershipColor();
	}

	public DefaultConfig(final DefaultConfig config) {
		this.obstacleColor = config.obstacleColor;
		this.sourceColor = config.sourceColor;
		this.targetColor = config.targetColor;
		this.targetChangerColor = config.targetChangerColor;
		this.absorbingAreaColor = config.absorbingAreaColor;
		this.densityColor = config.densityColor;
		this.stairColor = config.stairColor;
		this.pedestrianColor = config.pedestrianColor;
		this.measurementAreaColor = config.measurementAreaColor;
		this.measurementAreaAlpha = config.measurementAreaAlpha;
		initGroupMembershipColor();

		this.changed = config.changed;
	}

	/**
	 * Use this color palette: https://www.color-hex.com/color-palette/38840
	 * Watch out: The user cannot change these colors currently!
	 * */
	private void initGroupMembershipColor() {
		groupMembershipColors.put(GroupMembership.IN_GROUP, new Color(213,94,0));
		groupMembershipColors.put(GroupMembership.OUT_GROUP, new Color(0,0,0));
		groupMembershipColors.put(GroupMembership.OUT_GROUP_FRIENDLY, new Color(0,135,98));
		groupMembershipColors.put(GroupMembership.OUT_GROUP_NEUTRAL, new Color(153,153,153));
		groupMembershipColors.put(GroupMembership.OUT_GROUP_HOSTILE, new Color(229,229,0));
	}

	// Getter
	public synchronized boolean hasChanged() {
		return changed;
	}
	public Color getObstacleColor() {
		return obstacleColor;
	}
	public Color getSourceColor() {
		return sourceColor;
	}
	public Color getTargetColor() {
		return targetColor;
	}
	public Color getTargetChangerColor() {
		return targetChangerColor;
	}
	public Color getAbsorbingAreaColor() {
		return absorbingAreaColor;
	}
	public Color getDensityColor() {
		return densityColor;
	}
	public Color getStairColor() {
		return stairColor;
	}
	public Color getPedestrianColor() {
		return pedestrianColor;
	}
	public Color getMeasurementAreaColor() {
		return measurementAreaColor;
	}
	public int getMeasurementAreaAlpha() {
		return measurementAreaAlpha;
	}
	public Color getGroupMembershipColor(GroupMembership groupMembership) {
		return groupMembershipColors.get(groupMembership);
	}

	// Setter
	protected synchronized void setChanged() {
		this.changed = true;
	}

	public synchronized void clearChange() {
		changed = false;
	}

	public void setObstacleColor(final Color obstacleColor) {
		this.obstacleColor = obstacleColor;
		setChanged();
	}

	public void setSourceColor(Color sourceColor) {
		this.sourceColor = sourceColor;
		setChanged();
	}

	public void setTargetColor(final Color targetColor) {
		this.targetColor = targetColor;
		setChanged();
	}

	public void setTargetChangerColor(final Color targetChangerColor) {
		this.targetChangerColor = targetChangerColor;
		setChanged();
	}

	public void setAbsorbingAreaColor(final Color absorbingAreaColor) {
		this.absorbingAreaColor = absorbingAreaColor;
		setChanged();
	}

	public void setDensityColor(final Color densityColor) {
		this.densityColor = densityColor;
		setChanged();
	}

	public void setStairColor(final Color stairColor) {
		this.stairColor = stairColor;
		setChanged();
	}
	public void setPedestrianColor(Color pedestrianColor) {
		this.pedestrianColor = pedestrianColor;
	}

	public void setMeasurementAreaColor(Color measurementAreaColor) {
		this.measurementAreaColor = measurementAreaColor;
	}

	public void setMeasurementAreaAlpha(int measurementAreaAlpha) {
		this.measurementAreaAlpha = measurementAreaAlpha;
	}

}
