import nibabel as nib


def resove_bug(path):
    img = nib.load(path)
    qform = img.get_qform()
    img.set_qform(qform)
    sform = img.get_sform()
    img.set_sform(sform)
    nib.save(img, path)
