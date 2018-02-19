"""
ZetCode PyQt4 tutorial 

This example shows an icon
in the titlebar of the window.

author: Jan Bodnar
website: zetcode.com 
last edited: October 2011
"""

import sys
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.measure as msr
from skimage import segmentation as sgm
from skimage import morphology as mph
from skimage.filter import threshold_otsu, threshold_adaptive
import skimage.transform as tr 
from PyQt4 import QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends import qt4_compat
from BrightField_Segmentation_final import *


#use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE





class GUI(QtGui.QWidget):
    
    def __init__(self):

        self.FFimage=0;
        self.BCK=0;
        self.Std_Dev_image=[0,0];
        self.mask=[0,0]
        self.savepath=0
        self.BFpath=0
        self.randcmap = plt.matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))

        self.Hoffset=0 
        self.Voffset=0

        super(GUI, self).__init__()
        
        self.setWindowTitle('FROS ANALYSIS')

        self.initUI()
        
    #-----------------------------------------------------------------------------------------------
    # INITIALIZATION OF THE WINDOW - DEFINE AND PLACE ALL THE WIDGETS
    #-----------------------------------------------------------------------------------------------

    def initUI(self):
        


        # SET THE GEOMETRY
        mainWindow = QtGui.QVBoxLayout()
        self.setLayout(mainWindow)
        self.setGeometry(100, 100, 900, 900)
        
        IObox = QtGui.QHBoxLayout()
        Canvasbox = QtGui.QHBoxLayout()
        Actionsbox = QtGui.QHBoxLayout()
        Textbox = QtGui.QVBoxLayout()


        Spinboxes = QtGui.QGridLayout()

        mainWindow.addLayout(IObox)
        mainWindow.addLayout(Canvasbox)
        mainWindow.addLayout(Actionsbox)
        mainWindow.addLayout(Textbox)

        Canvasbox.addLayout(Spinboxes)



#        mainWindow.setRowMinimumHeight(9, 100)
#        mainWindow.setRowMinimumHeight(12, 100)

        # DEFINE ALL WIDGETS AND BUTTONS
        

        #I/O
        loadBtn = QtGui.QPushButton('Load Bright Field image')
        loadBCKbtn = QtGui.QPushButton('Load Background Image')
        saveBtn = QtGui.QPushButton('Save data')

        #ACTIONS
        
        st_devBtn=QtGui.QPushButton('Compute Std Dev Image')
        SegmentationBtn=QtGui.QPushButton('Compute Segmentation Mask')
        remove_oddshsapeBtn=QtGui.QPushButton('Remove odd shaped objects')
        ALLSegmentationBtn=QtGui.QPushButton('Do ALL segmentation')


        ThScLbl = QtGui.QLabel('Threshold scaling:')
        SeedMinLbl = QtGui.QLabel('Seeds minimum size:')
        SeedMaxLbl = QtGui.QLabel('Seeds maximum size:')
        CellMinLbl = QtGui.QLabel('Cells minimum size:')
        CellMaxLbl = QtGui.QLabel('Cells maximum size:')
        OffStLbl = QtGui.QLabel('Offset:')
        BlckSzLbl = QtGui.QLabel('Block Size:')
            
        #SPINBOXES
        self.ThSc = QtGui.QDoubleSpinBox(self)
        self.ThSc.setSingleStep(0.01)
        self.ThSc.setMaximum(10000)
        self.ThSc.setValue(1)
        

        self.SeedMin = QtGui.QSpinBox(self)        
        self.SeedMin.setMaximum(100000)
        self.SeedMin.setValue(0)
        
        self.SeedMax = QtGui.QSpinBox(self)        
        self.SeedMax.setMaximum(100000)
        self.SeedMax.setValue(500)        

        self.CellMin = QtGui.QSpinBox(self)
        self.CellMin.setMaximum(100000)
        self.CellMin.setValue(50)
                
        self.CellMax = QtGui.QSpinBox(self)
        self.CellMax.setMaximum(100000)    
        self.CellMax.setValue(500)

        self.OffSt = QtGui.QSpinBox(self)
        self.OffSt.setRange(-100000,100000)    
        self.OffSt.setValue(-300)

        self.BlckSz = QtGui.QSpinBox(self)
        self.BlckSz.setMaximum(100000)    
        self.BlckSz.setValue(101)        

        self.StdBox = QtGui.QCheckBox('Method : Std Dev')
        self.OtsuBox = QtGui.QCheckBox('Method : Otsu')
        self.AdaptBox = QtGui.QCheckBox('Method : Adapt Threshold')
     
	


        self.fig1 = Figure((8.0, 8.0), dpi=100)
        self.fig1.subplots_adjust(left=0., right=1., top=1., bottom=0.)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setFixedSize(QtCore.QSize(400,400))
    
        self.fig2 = Figure((8.0, 8.0), dpi=100)
        self.fig2.subplots_adjust(left=0., right=1., top=1., bottom=0.)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setFixedSize(QtCore.QSize(400,400))

        
    

        # PLACE ALL THE WIDGET ACCORDING TO THE mainWindowS

        IObox.addWidget(loadBtn)
        IObox.addWidget(loadBCKbtn)
 
        IObox.addWidget(saveBtn)
        
        Canvasbox.addWidget(self.canvas1)
        Canvasbox.addWidget(self.canvas2)


        Actionsbox.addWidget(st_devBtn)
        Actionsbox.addWidget(remove_oddshsapeBtn)
        Actionsbox.addWidget(SegmentationBtn)
        Actionsbox.addWidget(ALLSegmentationBtn)


        self.Toprinttext = QtGui.QTextEdit()
        Textbox.addWidget(self.Toprinttext)

        mainWindow.addWidget(self.HLine())
        
        Spinboxes.addWidget(ThScLbl,0,0,1,1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.ThSc,0,1,1,1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(SeedMinLbl,1,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.SeedMin,1,1, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(SeedMaxLbl,2,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.SeedMax,2,1, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(CellMinLbl,3,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.CellMin,3,1, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(CellMaxLbl,4,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.CellMax,4,1, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.AdaptBox,5,0,1,1)
        Spinboxes.addWidget(self.OtsuBox,6,0,1,1)
        Spinboxes.addWidget(self.StdBox,7,0,1,1)

        Spinboxes.addWidget(self.HLine(),8,0,1,1,QtCore.Qt.AlignTop)   

        Spinboxes.addWidget(OffStLbl,9,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.OffSt,9,1, 1, 1, QtCore.Qt.AlignTop)             

        Spinboxes.addWidget(BlckSzLbl,10,0, 1, 1, QtCore.Qt.AlignTop)
        Spinboxes.addWidget(self.BlckSz,10,1, 1, 1, QtCore.Qt.AlignTop)             

        mainWindow.addWidget(self.HLine())
                
        mainWindow.addWidget(self.VLine())


        
        

        self.setFocus()
        self.show()
        
        # BIND BUTTONS TO FUNCTIONS
        
        loadBtn.clicked.connect(self.loadDataset)
        loadBCKbtn.clicked.connect(self.loadBCKimage)
        saveBtn.clicked.connect(self.saveMask)
        SegmentationBtn.clicked.connect(self.Compute_SegmentationMask)
        remove_oddshsapeBtn.clicked.connect(self.Remove_odd_shaped_objects)
        st_devBtn.clicked.connect(self.Compute_StdDev)
        ALLSegmentationBtn.clicked.connect(self.do_ALL_segmentation)

        self.fig2.canvas.mpl_connect('button_press_event', self.remove_cells)


#        self.SeedMin.valueChanged.connect(self.updateCanvas1)

        
    #-----------------------------------------------------------------------------------------------
    # FORMATTING THE WINDOW
    #-----------------------------------------------------------------------------------------------

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def HLine(self):
        toto = QtGui.QFrame()
        toto.setFrameShape(QtGui.QFrame.HLine)
        toto.setFrameShadow(QtGui.QFrame.Sunken)
        return toto

    def VLine(self):
        toto = QtGui.QFrame()
        toto.setFrameShape(QtGui.QFrame.VLine)
        toto.setFrameShadow(QtGui.QFrame.Sunken)
        return toto

    
    #-----------------------------------------------------------------------------------------------
    # BUTTON FUNCTIONS
    #-----------------------------------------------------------------------------------------------

    def loadDataset(self):

        if self.BFpath:
            self.BFpath = QtGui.QFileDialog.getOpenFileName(self,'Choose an Image', self.BFpath)
        else:    
            self.BFpath = QtGui.QFileDialog.getOpenFileName(self,'Choose an Image', '.')
        self.BF = io.imread(str(self.BFpath))
        self.bfname = str(self.BFpath).split("/")[-1]

        self.updateCanvas1()
        self.print_on_textbox(self.BFpath)
        self.setFocus()


    def loadBCKimage(self):

        self.BCKpath = QtGui.QFileDialog.getOpenFileName()
        self.BCK = io.imread(str(self.BCKpath))
        plt.imshow(self.BCK)
        plt.show()


    def Compute_StdDev(self):

        self.Std_Dev_image = std_image(self.BF)

    def Remove_odd_shaped_objects(self):
      
        for i in range(self.mask2.max()):

            tmpmask=self.mask2==i        
            rps=msr.regionprops(tmpmask.astype(np.int8))
            
            if rps[0].eccentricity<0.85:
                self.mask2[self.mask2==i]=0
        [l,n]=msr.label(self.mask2,return_num=True)
        remove_reg(l,50,1000)
        self.mask2=l
        self.updateCanvas1()
        self.updateCanvas2()


    def Compute_SegmentationMask(self):

        
        if self.StdBox.isChecked():

            if sum(sum(self.Std_Dev_image)) == 0:
                QtGui.QMessageBox.warning(self, "Cannot perform required operation",
                "Compute the standard deviation image first!",
                QtGui.QMessageBox.Ok)

                return
            else:
                method='std'

        if self.AdaptBox.isChecked():

            method = 'adaptive'

        if self.OtsuBox.isChecked():

            method = 'otsu'


        [s,n,l] = mySegmentation(self.BF, self.Std_Dev_image, method, 'B' ,self.ThSc.value(), self.CellMin.value(), self.CellMax.value(),
                                 self.SeedMin.value(), self.SeedMax.value(), self.BlckSz.value(), self.OffSt.value())
        self.s=s
        self.mask=l
        self.mask2 = np.copy(self.mask)
        self.print_on_textbox(str(n))
        self.updateCanvas2()
        self.updateCanvas1()

    def do_ALL_segmentation(self):

        self.ALLpath = QtGui.QFileDialog.getExistingDirectory(self,'choose folder containing all the bright field images','H:\\Jacopo')
        self.savepath = QtGui.QFileDialog.getExistingDirectory(self,'choose folder to save masks','D:\\ANALISYS\\2015\\1.CLUSTER_OF_RECEPTORS_TRACKING')

        files=glob.glob(self.ALLpath + '/*.tif')
        print(files)
        for i in files:
            self.BF=io.imread(i)
            self.bfname = str(i).split("\\")[-1]

            self.Compute_SegmentationMask()
            self.Remove_odd_shaped_objects()
            
            myreply=QtGui.QMessageBox.warning(self, "are you ok with this segmentation?","go on?",QtGui.QMessageBox.Ok, QtGui.QMessageBox.No,QtGui.QMessageBox.Cancel)             

            if myreply==QtGui.QMessageBox.Ok:
                io.imsave(str(self.savepath) + '/' + self.bfname[:-4] + '_mask' + '.tif', self.mask2)                
            if myreply==QtGui.QMessageBox.Cancel:
                break






    def saveMask(self):

        
        self.savepath = QtGui.QFileDialog.getExistingDirectory(self,'choose saving position','D:\\ANALISYS\\2015\\1.CLUSTER_OF_RECEPTORS_TRACKING\\1.Tracks_and_masks_for_mobility_structures')
                   

        io.imsave(str(self.savepath) + '/' + self.bfname[:-4] + '_mask' + '.tif', np.uint16(self.mask2))


    def remove_cells(self, event):

        value=self.mask2[np.round(event.ydata),np.round(event.xdata)]

        self.mask2[self.mask2==value]=0      
        self.updateCanvas2()
        self.updateCanvas1()        

    #-----------------------------------------------------------------------------------------------
    # UPDATE FUNCTIONS
    #-----------------------------------------------------------------------------------------------

    def print_on_textbox(self,mystring):

        self.Toprinttext.append(mystring)

    def updateCanvas1(self):
        self.ax1.cla()
        if sum(sum(self.mask)) !=0:
            
            self.BF2=np.copy(self.BF)
            borders=sgm.find_boundaries(self.mask2)
            self.BF2[borders]=0.2*np.double(self.BF2.max())
            self.ax1.imshow(self.BF2, cmap = 'gray', interpolation='none')
            self.canvas1.draw()
            self.setFocus()
        else:

            self.ax1.imshow(self.BF, cmap = 'gray', interpolation='none')#,vmax=mymin*15,vmin=mymin)
            self.fig1.subplots_adjust(left=0., right=1., top=1., bottom=0.)
            self.canvas1.draw()
            self.setFocus()

    def updateCanvas2(self):
        cmap=self.randcmap
        cmap.set_under(color='white')
        self.ax2.cla()
        self.ax2.imshow(self.mask2, cmap = cmap, interpolation='none',vmin=0.0001)#,vmax=mymin*15,vmin=mymin)
        self.fig2.subplots_adjust(left=0., right=1., top=1., bottom=0.)
        self.canvas2.draw()
        self.setFocus()        
     
        
    def keyPressEvent(self, event):
#        print(event.key())

        self.mask2=self.mask2.astype(np.float64)
        self.mask=self.mask.astype(np.float64)

        if event.key() == QtCore.Qt.Key_Right:

            self.Hoffset-=1
            self.tform = tr.SimilarityTransform(translation=(int(self.Hoffset),int(self.Voffset)))
            self.mask2 = tr.warp(self.mask, self.tform,output_shape = (int(512), int(512)))
            [self.mask2,n]=msr.label(self.mask2,return_num=True) 
            self.mask2=sgm.clear_border(self.mask2)       
            self.updateCanvas2()
            self.updateCanvas1()

        if event.key() == QtCore.Qt.Key_Left:
            
            self.Hoffset+=1

            self.tform = tr.SimilarityTransform(translation=(self.Hoffset,self.Voffset))
            self.mask2 = tr.warp(self.mask, self.tform,output_shape = (int(512), int(512)))
            [self.mask2,n]=msr.label(self.mask2,return_num=True)
            self.mask2=sgm.clear_border(self.mask2)
            self.updateCanvas2()
            self.updateCanvas1()

        if event.key() == QtCore.Qt.Key_Up:
            
            self.Voffset+=1

            self.tform = tr.SimilarityTransform(translation=(self.Hoffset,self.Voffset))
            self.mask2 = tr.warp(self.mask, self.tform,output_shape = (int(512), int(512)))
            [self.mask2,n]=msr.label(self.mask2,return_num=True)
            self.mask2=sgm.clear_border(self.mask2)
            self.updateCanvas2()
            self.updateCanvas1()
            
        if event.key() == QtCore.Qt.Key_Down:
            
            self.Voffset-=1

            self.tform = tr.SimilarityTransform(translation=(self.Hoffset,self.Voffset))
            self.mask2 = tr.warp(self.mask, self.tform,output_shape = (int(512), int(512)))
            [self.mask2,n]=msr.label(self.mask2,return_num=True)
            self.mask2=sgm.clear_border(self.mask2)
            self.updateCanvas2()
            self.updateCanvas1()

        self.setFocus()
            


if __name__ == '__main__':
    
    app = QtGui.QApplication.instance() # checks if QApplication already exists 
    if not app: # create QApplication if it doesnt exist 
        app = QtGui.QApplication(sys.argv)
    
    app.setStyle("mac")
    gui = GUI()
    sys.exit(app.exec_())

###############################################################################
#        # WILL BE IMPORTANT FOR MOUSE EVENTS!!!!!!!!!!!!!!!
#        print(self.canvas1.underMouse())
###############################################################################
        

import scipy as sp
from scipy import optimize
import numpy as np

from scipy import ndimage

import glob
import os as os
from pylab import *
from skimage import morphology as mph
from skimage.filter import threshold_otsu, threshold_adaptive
from skimage import measure as msr
from skimage import segmentation as sgm

import matplotlib.pyplot as plt

def mySegmentation(img,s,method='adaptive',BoW='B',thr=0.75,l_th1=0,l_th2=550,seeds_thr1=50,seeds_thr2=500,block_size=7,offs=-0,visual=0):
    
    """the user can choose wether to use otsu for seeds (merkers) definition or get seeds from the standard deviation map"""
    img=Prepare_im(img);  
    sz=np.shape(img)
    seeds=np.zeros((sz[0],sz[1]))

    if method=='otsu':


        t=threshold_otsu(img.astype(uint16))*thr
        l=img<t

                #seeds=abs(seeds-1)      

        [l,N]=msr.label(l,return_num=True)

        [l,N]=remove_reg(l,l_th1,l_th2)   



    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
        
    if method=='adaptive':


        binary_adaptive = threshold_adaptive(-img, block_size, offset=offs)

        l=binary_adaptive

        [l,N]=msr.label(l,return_num=True)

        l,N=remove_reg(l,l_th1,l_th2)


        l=l!=0

        l=sgm.clear_border(l)

        l=mph.dilation(l)


        [l,n]=msr.label(l,return_num=True)

    if method=='std' :

        s2=np.copy(s)
        #s=std_image(img,0)
        t=mht.otsu(img.astype(uint16))*thr
        tempseeds=img<t
        s2[tempseeds==0]=0

        #seeds=pymorph.regmin((s2).astype(np.uint16)) #,array([[False,True,True,False],[False,True,True,False]]))
        seeds=abs(seeds-1)

        #seeds,n=mht.label(seeds)
        

        [seeds,N]=mht.label(seeds)
        seeds,N=remove_reg(seeds,seeds_thr1,seeds_thr2)   

        [seeds,N]=mht.label(seeds) 
    
        l = mht.cwatershed(img, seeds)
        #l=mph.watershed(img,seeds)
        l=l.astype(int32)        
        l,n=remove_reg(l,l_th1,l_th2)
        l=mht.labeled.remove_bordering(l)
        print 'label'
        print mht.labeled.labeled_size(l)

    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
    return seeds,N,l
              

    if visual:
        figure();imshow(img)
        figure();imshow(seeds)
        figure();imshow(l)
        
    return seeds,N,l

        
#############################################################################    
"""functions used by many other functions"""
#############################################################################


    

"""Fitting the plane to subract"""
def myplane(p,x,y):

    return p[0]*x+p[1]*y+p[2] 
    
def res(p,data,x,y):
    
    a=(data-myplane(p,x,y));
    
    return array(np.sum(np.abs(a**2)))
    
def fitplane(data,p0):
    
    s=np.shape(data);
    
    [x,y]=meshgrid(arange(s[1]),arange(s[0]));
    #p=optimize.leastsq(res,p0,args=(data,x,y),factor=1,maxfev=1000000);
    p=optimize.fmin(res,p0,args=(data,x,y));
    return p

    
def createDisk( size ):
    x, y = np.meshgrid( np.arange( -size, size ), np.arange( -size, size ) )
    diskMask = ( ( x + .5 )**2 + ( y + .5 )**2 < size**2)
    return diskMask
 

# =prepare the image for watershed transform correcting for inhomogeneous illumination 
def Prepare_im(img):
        
    s=np.shape(img)    
    p0=np.array([0,0,0])
    
    p0[0]=(img[0,0]-img[0,-1])/s[0]    
    p0[1]=(img[1,0]-img[1,-1])/s[1]
    p0[2]=img.mean()
    
    [x,y]=np.meshgrid(np.arange(s[1]),np.arange(s[0]))
    
    p=fitplane(img,p0)    
    img=img-myplane(p,x,y)    
    
    img=ndimage.gaussian_filter(img,1)
    m=img.min()
    img=img-m
    #img=abs(img)
    
    img=img.astype(uint16)
    
    
    return img

"""compute and returns the standard deviation image of an image"""
def std_image(img,visual=0,subs=2):
    
    img=ndimage.gaussian_filter(img,3)
    img_std = ndimage.filters.generic_filter(img, np.std, size=subs);
    if visual:    
        imshow(img_std)
    
    return img_std

        
        
def remove_reg(l,th1,th2):        
    """l must be a labeled image"""



    for i in range(l.max()):
        
        label_size=(l==i).sum()

        if (label_size>th2):
            l[l==i]=0
        if (label_size<th1):
            l[l==i]=0

    [l,N]=msr.label(l,return_num=True)
                   
    return l,N
    
    


        
        

    


