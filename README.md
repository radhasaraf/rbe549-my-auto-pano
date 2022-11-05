# AutoPano - Panorama Stitcher
- Stitches two or more images to create a seamless panorama image using classical &amp; deep learning approaches. 
- This is part of the course [RBE549:Classical and deep learning approaches to computer vision](https://nitinjsanket.github.io/teaching/rbe549/fall2022.html)

<!--
<table>
  <tr>
    <td>First Screen Page</td>
     <td>Holiday Mention</td>
     <td>Present day in purple and selected day in pink</td>
  </tr>
  <tr>
    <td valign="top"><img src="Phase1/Data/Test/TestSet2/1.jpg"></td>
    <td valign="top"><img src="Phase1/Data/Test/TestSet2/1.jpg"></td>
    <td valign="top"><img src="Phase1/Data/Test/TestSet2/1.jpg"></td>
  </tr>
 </table>
-->

# Results 

### Test Set
Given Images
<p float="middle">
  <img src="Phase1/Data/Test/TestSet2/1.jpg" width="250" hspace="20" />
  <img src="Phase1/Data/Test/TestSet2/2.jpg" width="250" hspace="20" /> 
  <img src="Phase1/Data/Test/TestSet2/3.jpg" width="250" hspace="20" />
</p>

Output
<p float="middle">
<img src="report/phase1/testset2_image123_clear_stitch3.png" width="500" height="500"/>
</p>

### Custom Set
Following are the images we took at WPI

<p float="middle">
  <img src="Phase1/Data/Train/CustomSet1/1.jpg" width="300" hspace="20" />
  <img src="Phase1/Data/Train/CustomSet1/2.jpg" width="300" hspace="20" /> 
</p>

Output
<p float="middle">
<img src="report/phase1/customset1_image123_clear_stitch3.png" width="700" height="500"/>
</p>

## Design Overview

### Recognizing the panorama
We did a brute force feature matching with all possible pairs and recognised the best possible way to stitch images, aka recognizing panorama, using graph based approach outlined below

<p float="middle">
<img src="report/phase1/graph_appraoch.png" />
</p>


## Collaborators 
Radha Saraf - rrsaraf@wpi.edu

Ramana - spinnamaraju@wpi.edu
