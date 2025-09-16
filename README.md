v3.0
- For evaluating best-landmark's accuracy, "triangulate3DHPE_bestmp_accuracy.py" and "triangulate3DHPE_bestmp_accuracy_conf90_perframe.py" was made
    -  "triangulate3DHPE_bestmp_accuracy.py" calcurate MPJPE between best-landmark and detaset's landmark. you can determine the threshold of reliability. it omit a frame which exist dataset's landmark that its reliability is lower then it
    - "triangulate3DHPE_bestmp_accuracy_conf90_perframe.py" culculate MPJPE between best-landmark and dataset's landmark which is not lower than reliability threshold. (This one omit only landmark. not frame)
- Culculation body's MPJPE, "compute_mpjpe.py" was made. 
- "compute_mpjpe_copy.py" in production...

v2.0
- "triangulate3DHPE_partial_accuracy.py" can now display MPJPE for both hands and feet for each frame
- Added "best_landmarks.json" which saves the most reliable landmarks for each landmark in each frame across all views, and "triangulate3DHPE_mpdata.py" which generates it.
- Changed the format of estimated landmark data in "triangulate3DHPE_data.py"
    - Changed from [x0, y0, z0, x1, ... , z32] to {"0":[x0, y0, z0], "1":[x1, y1, z1], ... , "32":[x32, y32, z32]} dictionary format


v1.0
- add "triangulate3DHPE_data.py", "triangulate3DHPE_show_result.py", "triangulate3DHPEcompute_mpjpe.py", "triangulate3DHPE_partial_accuracy"
    - triangulate3DHPE_data aggregate predicted landmarks
    - triangulate3DHPE_show_result show 3Dplots each frame
    - triangulate3DHPEcompute_mpjpe compute mpjpe between GT landmark and predicted landmark
    - triangulate3DHPE_partial_accuracy evaluate the accuracy of landmarks not present in the GT data (incomplete)

v0.2
- make concatenated images video using only 2 video

v0.1
- Be able to save annotate image and non-annotate image
- add "concatenation_images" function    
    - save and concatenate output images
- collect related files into "related_files" folder added

v0.0
- capture HD video
- dataset's internal params and external params input
- adjusted 3d-plot view init
- save 3d-plot