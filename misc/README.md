## Miscellaneous
| File       | Description                          |
   |------------|--------------------------------------|
   | select_keypoint.py   | Select keypoint on mesh. |
   | vis_keypoint.py   | Visualize keypoint annotations. |

### select_keypoint.py

Use this Python script to select keypoints on a mesh. Currently supports the following arguments

- --urdf_file (.urdf)

```
python misc/select_keypoint.py --urdf_file assets/articulated_objs/backup/backup/box/mobility.urdf
```

    ⚠️ WARNING ⚠️
    
    When selecting a keypoint it will overwrite the existing keypoints.json in the object's asset directory. Please move the existing keypoints.json file to another location if you wish to preserve the points.


To select keypoints, **right click** on the mesh where you'd like the keypoint to appear. A dot will appear to signify where the keypoint is annotated. If you do not like the placement of the keypoint **left click** to remove annotated keypoint. At the end all keypoints are saved to a `keypoints.json` file under the same parent directory of the asset you've provided. 

Using the example as above the following would occur:

```
articulated_objs/
├── box/
│   ├── keypoints.json
│   └── mobility.urdf
```

### vis_keypoint.py

Use this Python script to select keypoints on a mesh. Currently supports the following arguments as input

- --mesh_file (.obj) 
- --urdf_file (.urdf)
- --ply_file (.ply)



An example of a runnable script:
```
python misc/vis_keypoint.py --urdf_file assets/articulated_objs/window/102801_link_0/mobility.urdf
```
We assume there is a keypoints.json file located under the parent directory of the mesh file. Here's an example using the command from above:

```
window/
├── 102801_link_0/
│   ├── keypoints.json
│   └── mobility.urdf
```