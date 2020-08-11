import xml.etree.ElementTree as ET
import re
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
from distutils.dir_util import copy_tree
import os

# Coefficient nomenclature consistent with: 
#	Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems, by Brian L. Stevens et al., Wiley, 2016.

def parse_pose(pose_str):
	splits = re.findall(r'[-+]*[0-9]+\.*[0-9]*e*[+\-]*[0-9]*',pose_str)
	pose_vals = [float(num_str) for num_str in splits]

	pos = np.array(pose_vals[0:3])
	rpy = np.array(pose_vals[3:6])

	return pos, rpy	

# Open model based on script's calling argument
model_name = sys.argv[1]
tree = ET.parse(model_name + '/' + model_name + '.sdf')
root = tree.getroot()

# The coefficients for the whole aircraft will be calulated about the fuselage center of gravity
cog_str = root.find(".//link[@name='fuselage']//pose").text
cog_pos = parse_pose(cog_str)[0]

model = root.find(".//model")
airfoils = model.findall(".//plugin[@filename='libLiftDragPlugin2.so']")

# Assume that a few properties of the left wing apply to the whole vehicle
wing = root.find(".//plugin[@name='wing_left']")
cl0_wing = float(wing.find(".//cL0").text)
cd0_wing = float(wing.find(".//cD0").text)
alpha_stall = wing.find(".//alpha_stall").text
cla_stall = wing.find(".//cLa_stall").text
cma_stall = wing.find(".//cma_stall").text
kcDcL = float(wing.find(".//kcDcL").text)
air_density = float(wing.find(".//air_density").text)


# Compute each airfoil's effect on the whole aircraft
tot_cla = 0
tot_cma = 0
tot_cyb = 0
tot_cnb = 0
tot_crb = 0
tot_crp = 0
tot_cl0 = 0
tot_cm0 = 0
tot_area = 0
for airfoil in airfoils:
	cla_foil = float(airfoil.find(".//cLa").text)
	cma_foil = float(airfoil.find(".//cma").text)
	cl0_foil = float(airfoil.find(".//cL0").text)
	cm0_foil = float(airfoil.find(".//cm0").text)
	area = float(airfoil.find(".//area").text)
	pose_str = airfoil.find(".//pose").text
	pos, rpy = parse_pose(pose_str) # Get position and orientation of airfoil in the fuselage

	# Rotate important vectors from airfoil frame into fuselage frame
	rot = R.from_euler('xyz', rpy, degrees=False) # Note: Gazaebo model files use euler angles roll, pitch, yaw to describe link pose
	fwdI = rot.apply([1,0,0]) # Airfoil's forward direction in fuselage coordinates
	spanI = rot.apply([0,-1,0]) # Wing's span direction in fuselage coordinates
	upI = rot.apply([0,0,1]) # Airfoil's upward direction in fuselage coordinates

	# Calculate effects of rotated airfoil's lift and moment on the whole aircraft
	cla_fus = cla_foil*abs(np.dot(spanI,[0,-1,0]))
	cma_fus = cma_foil*abs(np.dot(spanI,[0,-1,0]))
	cyb_fus = cla_foil*abs(np.dot(spanI,[0,0,1]))
	cnb_fus = cma_foil*abs(np.dot(spanI,[0,0,1]))

	x_foil_plane = [1,0,0] - np.dot(spanI,[1,0,0])*spanI # First component of the fuselage coordinates projected onto airfoil's lift, drag plane
	x_foil_plane = x_foil_plane/np.linalg.norm(x_foil_plane) # Normalize this vector to make dot product simple

	# Correct for wing installment angles on fuselage
	alpha_offset = np.arccos(np.dot( fwdI , x_foil_plane ))

	if np.dot( np.cross(x_foil_plane,fwdI) , spanI ) < 0: # Check if AoA is +/-. Note: spanI is assumed to be direction of positive pitch for airfoil, likewise [0,-1,0] is positive pitch in fuselage.
		alpha_offset = -alpha_offset

	cl0_foil = (cl0_foil + cla_foil*alpha_offset)
	cm0_foil = (cm0_foil + cma_foil*alpha_offset)
	cl0_fus =  cl0_foil*abs(np.dot(spanI,[0,-1,0]))
	cm0_fus =  cm0_foil*abs(np.dot(spanI,[0,-1,0]))
	# cy0_fus = cl0_foil*abs(np.dot(upwardI,[0,1,0])) # We assume these are 0 due to longitudinal symmetry
	# cr0_fus = cm0_foil*abs(np.dot(upwardI,[0,1,0])) # We assume these are 0 due to longitudinal symmetry

	# Move rigid body's force, moment pair to the fuselage cg
	r_cog_cp = pos - cog_pos
	cma_fus += r_cog_cp[0]*cla_fus
	cnb_fus += r_cog_cp[0]*cyb_fus
	cm0_fus += r_cog_cp[0]*cl0_fus
	# print('pos_cg, pos_cp: ' + str(cog_pos) + ', ' + str(pos))
	# print('r_cog_cp ' + str(r_cog_cp))
	# print('cl0_fus %f' % cl0_fus)
	# print('adding to cm0 %f' % (r_cog_cp[0]*cl0_fus))

	# roll damping derivative (roll moment coefficient contribution due to roll rate)
	roll_arm = r_cog_cp
	roll_arm[0] = 0
	roll_armI = roll_arm/np.linalg.norm(roll_arm) 
	crp_fus = -2*cla_foil*abs(np.dot(spanI,roll_armI))*np.linalg.norm(roll_arm)**2
	
	# dihedral derivative
	crb_fus = -cyb_fus*r_cog_cp[2]

	# Add weighted coefficients for this wing to the running total of the aircraft
	tot_cla += cla_fus*area
	tot_cma += cma_fus*area
	tot_cyb += cyb_fus*area
	tot_cnb += cnb_fus*area
	tot_cl0 += cl0_fus*area
	tot_cm0 += cm0_fus*area
	tot_crp += crp_fus*area
	tot_crb += crb_fus*area
	tot_area += area

	# Remove this wing from tree datastructure
	model.remove(airfoil)

cla = tot_cla/tot_area
cma = tot_cma/tot_area
cyb = tot_cyb/tot_area
cnb = tot_cnb/tot_area
cl0 = tot_cl0/tot_area
cm0 = tot_cm0/tot_area
crp = tot_crp/tot_area
crb = tot_crb/tot_area

# Place element in tree for the liftdrag plugin that summarize whole vehicle aerodynamics
plugin = ET.SubElement(root.find(".//model[@name='" + model_name + "']"),'plugin')
plugin.attrib['name'] = 'liftdragplug'
plugin.attrib['filename'] = 'libLiftDragPluginWhole.so'
plugin.tail = '\n'

def setsub(key,value):
	sub = ET.SubElement(plugin,key)
	sub.text = value
	sub.tail = '\n'	

setsub('link_name','fuselage')
setsub('cla',str(cla))
setsub('cl0',str(cl0))
setsub('cd_a',str(kcDcL))
setsub('cd_b',str(-2*cl0_wing*kcDcL))
setsub('cd_c',str(cd0_wing + kcDcL*cl0_wing**2))
setsub('cma1',str(cma))
setsub('cma0',str(cm0))
setsub('cnb',str(cnb))
setsub('crp',str(crp))
setsub('crb',str(crb))
setsub('chord',str(1))
setsub('alpha_stall',alpha_stall)
setsub('cma_stall',cma_stall)
setsub('cla_stall',cla_stall)
setsub('chord',str(1))
setsub('area',str(tot_area))
setsub('air_density',str(air_density))
setsub('forward','1 0 0')
setsub('upward','0 0 1')
setsub('left_elevon_name','aileron_left_joint')
setsub('right_elevon_name','aileron_right_joint')
setsub('elevator_name','elevator_joint')
setsub('rudder_name','rudder_joint')
setsub('verbose',str(1))

# Save as a new model
new_name = model_name + '_whole'
copy_tree(model_name, new_name)
tree.write(new_name + '/' + new_name + '.sdf')

# Remove old config
os.remove(new_name + '/' + model_name + '.sdf')

# Modify new model's config to point to new .sdf
tree = ET.parse(new_name + '/' + 'model.config')
root = tree.getroot()

sdf = root.find('.//sdf')
sdf.text = new_name + '.sdf'

description = root.find('.//description')
description.text = description.text + '\n__AUTOGENERATED__ from \'' + model_name + '\' by ' + sys.argv[0] + ' for configuration with libLiftDragPluginWhole.\
 Adapted from an .sdf utilizing libLiftDragPlugin2.\
 Some aerodynamic behavior may be lost to generality.\n'
tree.write(new_name + '/model.config')