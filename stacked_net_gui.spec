# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['stacked_net_gui.py'],
             pathex=['/Users/kirstennicassio/Desktop/JT'],
             binaries=[('/System/Library/Frameworks/Tk.framework/Tk','tk'),('/System/Library/Frameworks/Tcl.framework/Tcl','tcl')],
             datas=[],
             hiddenimports=['tkinter','tensorflow'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='stacked_net_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
