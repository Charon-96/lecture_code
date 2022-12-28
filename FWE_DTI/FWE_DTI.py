'''
FWE-DTI for single b value 


input: (FSL prepared files)
    DWI_eddy.nii.gz
    DWI.bval
    DWI_rotated.bvec
    b0_brain_mask.nii.gz
    
output：
    DTI_FA.nii
    DTI_MD.nii
    FWE-DTI_FA.nii
    FWE-DTI_MD.nii
    FWE-DTI_FW.nii     
    
'''

from pathlib import Path
import numpy as np
import nibabel as nib
import os
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.fwdti import FreeWaterTensorModel
from beltrami import BeltramiModel
from matplotlib import pyplot as plt


print('Loading data...')


filepath = R"D:\SF\a"  # prepared file dir
filenames = os.listdir(filepath)


n=len(filenames)
bek=[]

#exclued other files generated from prepared programs
for i in range(n):
    if ".txt" in filenames[i]:
        bek.append(i)
    if ".sh" in filenames[i]:
        bek.append(i)
   if ".rar" in filenames[i]:
        bek.append(i)
filenames = [filenames[i] for i in range(n) if (i not in bek)]

for name in filenames:
    print(name)
    path_now = filepath + '\\' + name
    if os.path.isdir(os.path.join(path_now,"results")):
        print(path_now);
    else:
        fdwi = path_now + '\\' + 'DWI_eddy.nii.gz'
        fbvals = path_now + '\\' + 'DWI.bval'
        fbvecs = path_now + '\\' + 'DWI_rotated.bvec'
        fmask = path_now + '\\' + 'b0_brain_mask.nii.gz'


        bvals, bvecs = read_bvals_bvecs(str(fbvals), str(fbvecs))
        data = nib.load(str(fdwi)).get_data()
        
    
    
        '''single-shell'''
        #if (acquired only with b = 1000) 
        bval_mask = np.logical_or(bvals == 0, bvals > 990)
        print(len(bval_mask))
        data_single = data[:, :, :, bval_mask]
        print(len(data_single))
        print(data_single.shape)
        bvals_single = bvals * 10**-3[bval_mask] * 10**-3  # rescaling bvals for Beltrami
        print(bvals_single)
        bvecs_single = bvecs#[bval_mask, :]
        #print(len(bvecs_single))
        gtab_single = gradient_table(bvals_single, bvecs_single, b0_threshold=0)
    
    
    
    
        '''
        # multi-shell(data with b = 200, 400 and 1000)
        bval_mask = bvals <= 1050
        #print(bval_mask)
        data_multi = data[:, :, :, bval_mask]
        #print(data_multi.shape)
        bvals_multi = bvals[bval_mask] * 10**-3  # rescaling bvals for Beltrami
        #print(len(bvals_multi))
        bvecs_multi = bvecs[bval_mask, :]
        gtab_multi = gradient_table(bvals_multi, bvecs_multi, b0_threshold=0)
        
        '''
    
    
    
    
    
        # loadind mask (previously computed with fsl bet)
        mask = nib.load(str(fmask)).get_data()
        mask = mask.astype(bool)
        #print(len(mask))
    
    
        # slicing data (for faster computation)
        data_single = data_single[17:89, 7:97, 36:56, :]   
        #data_multi = data_multi[17:89, 7:97, 36:56, :]
        mask = mask[17:89, 7:97, 34:54]
        print(len(mask))
    
        # masking data
        masked_single = data_single * mask[..., np.newaxis]
        #masked_multi = data_multi * mask[..., np.newaxis]

    
        '''unweighted image S0 '''
        #S0mul = np.mean(masked_multi[..., gtab_multi.b0s_mask], axis=-1) * mask
        S0sig = np.mean(masked_single[..., gtab_single.b0s_mask], axis=-1) * mask
    
    
    
        print('Running standard DTI (single shell)...')
        dtimodel = TensorModel(gtab_single)
        dtifit = dtimodel.fit(masked_single)
        dti_fa = dtifit.fa * mask
        dti_md = dtifit.md * mask
    
        dti_fw = np.zeros(dti_fa.shape) * mask
    
        os.makedirs(path_now + '\\' + 'results') #output dir
    
        affine = nib.load(str(fdwi)).affine
        dti_fa_img = nib.Nifti1Image(dti_fa, affine)
        nib.save(dti_fa_img, str(path_now + '\\' + 'results' + '\\' + 'Dti_fa.nii.gz'))
    
    
        dti_md_img = nib.Nifti1Image(dti_md, affine)
        nib.save(dti_md_img, str(path_now + '\\' + 'results' + '\\' + 'Dti_md.nii.gz'))
   
    
        WM = dti_fa > 0.7
        #CSF = dti_md > 2.5
        CSF = dti_md > 2.5
    
    
    
        St = np.round(np.percentile(S0[WM], 95))
        Sw = np.round(np.percentile(S0[CSF], 95))
        print('St = '+str(St))
        print('Sw = '+str(Sw))
    
  
       '''Beltami FW-DTI (single-shell)'''
        print('Running Beltami FW-DTI (single-shell)...')
        bmodel = BeltramiModel(gtab_single, init_method='hybrid', Stissue=St, Swater=Sw,
                               iterations=100, learning_rate=0.0005)
        bfit = bmodel.fit(masked_single, mask=mask)
        belt_fa_s = bfit.fa * mask
        belt_md_s = bfit.md * mask
        belt_fw_s = bfit.fw * mask
    
        dti_fa_img = nib.Nifti1Image(belt_fa_s, affine)
        nib.save(dti_fa_img, str(path_now + '\\' + 'results' + '\\' + 'Fwe_dti_fa.nii.gz'))
    
        dti_md_img = nib.Nifti1Image(belt_md_s, affine)
        nib.save(dti_md_img, str(path_now + '\\' + 'results' + '\\' + 'Fwe_dti_md.nii.gz'))
    
        dti_fw_img = nib.Nifti1Image(belt_fw_s, affine)
        nib.save(dti_fw_img, str(path_now + '\\' + 'results' + '\\' + 'Fwe_dti_fw.nii.gz'))
    
        print(name + '----over')
    
    
    
    
        '''Beltami FW-DTI (multi-shell)'''
        '''
        print('Running Beltami FW-DTI (multi-shell)...')
        bmodel = BeltramiModel(gtab_multi, init_method='hybrid', Stissue=St, Swater=Sw,
                               iterations=100, learning_rate=0.0005)
        bfit = bmodel.fit(masked_multi, mask=mask)
        belt_fa_m = bfit.fa * mask
        belt_md_m = bfit.md * mask
        belt_fw_m = bfit.fw * mask
        '''
        
        
        print('All done!')

