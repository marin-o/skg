from graph_creation.create_graph import create_nx_graph

if __name__ == '__main__':
    dataset_name = 'politeness'
    style_1_name = 'polite'
    style_2_name = 'neutral'

    graph = create_nx_graph(dataset_name, style_1_name, style_2_name)

    print("Created .tsv file")


