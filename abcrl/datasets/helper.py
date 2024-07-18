def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
