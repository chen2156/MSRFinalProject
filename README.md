# MSRFinalProject
Final Project for MSR

Implementation of this paper: https://ieeexplore.ieee.org/document/4543324

Goal of this paper is to be able to tell the distance of level objects from robot by only using cameras

This method involves implementing the Gaussian Process Algorithm

![Alt Text](20211104_112847.jpg)
<h1> Steps Taken to do the project</h1>

<h3>Hardware Setup</h3>

Turtlebot3
[additional waffle plate](https://www.robotis.us/tb3-waffle-plate-ipl-01-8ea/)
[M2 screws](https://www.amazon.com/HanTof-Washers-Assortment-Machine-Stainless/dp/B082XRX17Z/ref=asc_df_B082XRX17Z/?tag=hyprod-20&linkCode=df0&hvadid=416774286618&hvpos=&hvnetw=g&hvrand=16898008894177674308&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9021564&hvtargid=pla-901437054371&psc=1&tag=&ref=&adgrpid=95471660538&hvpone=&hvptwo=&hvadid=416774286618&hvpos=&hvnetw=g&hvrand=16898008894177674308&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9021564&hvtargid=pla-901437054371)
[convex mirror](https://www.edmundoptics.com/p/50mm-dia-x25mm-fl-enhanced-aluminum-convex-mirror-/29998/)
[M3 Standoffs](https://www.amazon.com/Csdtylh-Male-Female-Standoff-Stainless-Assortment/dp/B06Y5TJXY1/ref=sr_1_4?crid=1EYKXSMDMV6A7&dchild=1&keywords=m3+standoff+assortment&qid=1634152182&sprefix=m3+standoff+assortment%2Caps%2C443&sr=8-4)
Raspberry Pi camera
[3D printed camera mount](raspberreypi cameraholder.stl)

<h3>Building the robot</h3>
To build the robot, disassembled the top layer, including the LIDAR, place the Raspberry Pi camera on the second top layer such that it is centered on the robot.  Camera mount was 3D printed.  Move the USB2LDS board to the same layer as camera.  Assemble the waffle plate and attach the convex mirror to the bottom of it

