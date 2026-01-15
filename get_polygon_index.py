import bpy
import bpy_extras
import math
import os
import mathutils

file = open("polygon_index.txt", "w")

light_ups = [bpy.data.objects[f'light_up{i}'] for i in range(1, 11)]
light_bottoms = [bpy.data.objects[f'light_bottom{i}'] for i in range(1, 11)]
light_lefts = [bpy.data.objects[f'light_left{i}'] for i in range(1, 11)]
light_rights = [bpy.data.objects[f'light_right{i}'] for i in range(1, 11)]
light_flows = [bpy.data.objects[f'light_flow{i}'] for i in range(1, 11)]
fans = [bpy.data.objects[f'fan{i}'] for i in range(1, 11)]

mesh_light_up = light_ups[0].data
mesh_light_bottom = light_bottoms[0].data
mesh_light_left = light_lefts[0].data
mesh_light_right = light_rights[0].data
mesh_light_flow = light_flows[0].data
mesh_fan = fans[0].data

index_list = []

material_index_plastic = mesh_light_up.materials.find("plastic")
for polygon in mesh_light_up.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("up material_index_plastic \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_plastic = mesh_light_bottom.materials.find("plastic")
for polygon in mesh_light_bottom.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("bottom material_index_plastic \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_plastic = mesh_light_left.materials.find("plastic")
for polygon in mesh_light_left.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("left material_index_plastic \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_plastic = mesh_light_right.materials.find("plastic")
for polygon in mesh_light_right.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("right material_index_plastic \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_plastic = mesh_light_flow.materials.find("light_flow_plastic")
for polygon in mesh_light_flow.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("flow material_index_plastic \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_plastic = mesh_fan.materials.find("target_aim_(plastic)")
for polygon in mesh_fan.polygons:
    if polygon.material_index == material_index_plastic:
        index_list.append(polygon.index)
file.write("target_aim_(plastic) \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

#===================================================================================================
material_index_light_up = mesh_light_up.materials.find("light_up")
for polygon in mesh_light_up.polygons:
    if polygon.material_index == material_index_light_up:
        index_list.append(polygon.index)
file.write("material_index_light_up \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_light_bottom = mesh_light_bottom.materials.find("light_bottom")
for polygon in mesh_light_bottom.polygons:
    if polygon.material_index == material_index_light_bottom:
        index_list.append(polygon.index)
file.write("material_index_light_bottom \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_light_left = mesh_light_left.materials.find("light_left")
for polygon in mesh_light_left.polygons:
    if polygon.material_index == material_index_light_left:
        index_list.append(polygon.index)
file.write("material_index_light_left \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_light_right = mesh_light_right.materials.find("light_right")
for polygon in mesh_light_right.polygons:
    if polygon.material_index == material_index_light_right:
        index_list.append(polygon.index)
file.write("material_index_light_right \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_light_flow = mesh_light_flow.materials.find("light_flow")
for polygon in mesh_light_flow.polygons:
    if polygon.material_index == material_index_light_flow:
        index_list.append(polygon.index)
file.write("material_index_light_flow \n" + str(index_list) + "\n\n\n\n")
index_list.clear()

material_index_fan = mesh_fan.materials.find("target_aim_0_LED")
for polygon in mesh_fan.polygons:
    if polygon.material_index == material_index_fan:
        index_list.append(polygon.index)
file.write("target_aim_0_LED \n" + str(index_list) + "\n\n\n\n")