import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"], save_dir="/home/lanhf/INT3406-final-project/vncorenlp"
)

while True:
    text = input()
    print(" ".join(rdrsegmenter.word_segment(text)))
