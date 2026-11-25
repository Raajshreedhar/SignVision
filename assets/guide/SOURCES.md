# Sign guide photos: sources and license

The 29 photos in this folder are the hand-shape pictures in the app's sign guide, one per class in `config.CLASSES` order (A–Z, then SPACE, DELETE, NOTHING).

| | |
| --- | --- |
| Dataset | ASL Alphabet Test |
| Creator | Dan Rasband (Kaggle user `danrasband`) |
| URL | https://www.kaggle.com/datasets/danrasband/asl-alphabet-test |
| Version | 1, last updated 2018-08-01 |
| License | CC0 1.0 Universal public domain dedication (Kaggle license field: `CC0: Public Domain`), https://creativecommons.org/publicdomain/zero/1.0/ |

CC0 places the photos in the public domain, so they may be copied, changed and redistributed without conditions. Credit is given anyway, in the guide (`Photos: Dan Rasband, ASL Alphabet Test (CC0)`) and here.

## How the files were made

- Each file is a byte-identical copy of one dataset image, renamed to `<CLASS>.jpg`. Nothing was resized, cropped or re-encoded. Each image is a 200 × 200 JPEG.
- The files were copied from the kagglehub cache, `~/.cache/kagglehub/datasets/danrasband/asl-alphabet-test/versions/1/asl-alphabet-test/`. The source column below is relative to that folder.
- Selection rule: for each class, one frame that shows a single hand, no face, and the plainest background that class offers. For example, U0026 was chosen over U0015, which has a person at the edge, and B0030 over B0005, whose background is busy.
- This dataset is also the project's held-out classifier test set (`data/cls_test/`, never trained on). Showing 29 of its images in the guide changes no training or evaluation. The folder holds single renamed files, not a copy of the dataset.

`tests/test_guide_assets.py` checks every file against the SHA-256 in this table.

## Files

| File | Source (in the dataset) | SHA-256 |
| --- | --- | --- |
| A.jpg | A/A0005_test.jpg | 78ed23875740a490f29f23f2b9de37e747abf04222a5a11da803d50cb9842884 |
| B.jpg | B/B0030_test.jpg | 10f24cb13d155ef5115c6bef7a997967d4a390525a5551b6cb60a77c41a58d76 |
| C.jpg | C/C0025_test.jpg | 92589fa6eb9ecf2e7bf6ed7d9d33b28991f133e62a1efe699e81ee14eb888fdc |
| D.jpg | D/D0025_test.jpg | 2386a3f1f7895bd3a0b7f229ce737897dd4399cf33848a4c6cdfda2ba11e4e13 |
| E.jpg | E/E0020_test.jpg | df0685bd3121e179ddb31fb2c41635413d98b91e581202bca7439be62dbbaa5d |
| F.jpg | F/F0020_test.jpg | d8edd33f5c1cce8f7f3d7f030c735940003b688b898ac933a4c0fee0743f7c3e |
| G.jpg | G/G0015_test.jpg | b2b084839225cb320f73c5f2ba81a4ad65aa92aa873b5da239019b8043dc5b88 |
| H.jpg | H/H0025_test.jpg | 4514a6e6d25d2b40380361d198e1b3f758bf8ad12e2105f581fafbcc1a1665bb |
| I.jpg | I/I0020_test.jpg | 2374e08439b81a789cc030eb07c48a0d7a27e52b3089a878f430d8118a77178d |
| J.jpg | J/J0015_test.jpg | 9057f9bd566daa3ea21524addd1fd96b12495285f4fa84399e077eac9ca78e58 |
| K.jpg | K/K0025_test.jpg | e6a7792fd70040770836df80ac9a8d0e7c73d38cb87458e908bf1d11b686c850 |
| L.jpg | L/L0015_test.jpg | 5ca56429f1046a145760d81ed9ca96346c3a3ebc8efab72c45af8892c7313833 |
| M.jpg | M/M0015_test.jpg | 7f22aba9a8967f015e53bea03758ac668f577ad657b470b4f607e2e8899debbf |
| N.jpg | N/N0015_test.jpg | c8b59483c7e0e86161e94c8aeaab6d79ec3579c956fef4ac8bbbc01a64f04b9e |
| O.jpg | O/O0025_test.jpg | e45dae56f6215ce36546b7fe336ab87117779650c7fc84720dfaeae3bcba0503 |
| P.jpg | P/P0010_test.jpg | 0e79b5010966a0b8b01d2d9dce4008aab5294885ff74746b2a01e5c927e3c515 |
| Q.jpg | Q/Q0015_test.jpg | e1144596ba7c8fbed7f3d7a875ca0651b485efff8e84ccc3b6066368e9594f61 |
| R.jpg | R/R0025_test.jpg | f952899bc852bca54a6ea758a788b1cabf8d192b3d915a6224df504654dfd71e |
| S.jpg | S/S0025_test.jpg | 1a46c32eb25f61e622bba328f79d26bcf7f176b108a79d96b4b457a4ed794a14 |
| T.jpg | T/T0025_test.jpg | 46aba7c5a5276f2230feedf92a800b29e8e97961bda96b28bcd341e875c32741 |
| U.jpg | U/U0026_test.jpg | 9fbeadb0306f87f194ae432dd3a07f70c4e679d6789f313328ea082c12dc8412 |
| V.jpg | V/V0025_test.jpg | 104be6b4b4be69de10df4198207f8d8489c2900aba622be6446b96f106484dd7 |
| W.jpg | W/W0025_test.jpg | 92dc7e389e46c1fe19ffe220279f097111ff4405ee01a508ac00f0f942f36f29 |
| X.jpg | X/X0015_test.jpg | 3034241759913dab92c43543325cc8a047705805251a418f424d6dd55b549ea4 |
| Y.jpg | Y/Y0025_test.jpg | 111e815424c5a76bac009466d3a01de61e2e8fd23ac16dc19ca6a26c324e1d1f |
| Z.jpg | Z/Z0020_test.jpg | 0d226f89d076271fed1df15fc5aa4c72532cf08916256e5a3ee2a3bda1169078 |
| SPACE.jpg | space/space0015_test.jpg | fb9c910ed558edaf1f28c61c630898642fd0e79d30ffb0d75a996886e5c77ab7 |
| DELETE.jpg | del/del0020_test.jpg | f08992ccd10ab95de42de1c96ca40a572a342364568835e980dd4b6b2e2bd0b7 |
| NOTHING.jpg | nothing/nothing0025_test.jpg | 45035ccb7da7ca1e0d163913c5d1bab287d7cb399923a6ba4e1a0f54a89592e0 |
