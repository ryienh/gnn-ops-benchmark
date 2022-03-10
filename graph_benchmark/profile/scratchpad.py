# Define model
model = (
    GCNREG(input_dim=13, hidden_dim=20, dropout=0.1, num_conv_layers=5, heads=8)
    .double()
    .cuda()
)
# get dummy data #TODO: make random dataset class
tr_loader, _, _ = get_train_val_test_loaders(batch_size=64)
X = next(iter(tr_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train loop with pytorch profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    use_cuda=True,
) as prof_train:
    with record_function("model_train"):
        # single forward pass
        model = model.train()
        X = X.cuda()

        for _ in tqdm.tqdm(range(10)):
            optimizer.zero_grad()

            prediction = model(X)
            prediction = torch.squeeze(prediction)
            loss = model.loss(prediction, X.y, -1)

            # single backward pass
            loss.backward()
            optimizer.step()

# inf loop with pytorch profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    use_cuda=True,
) as prof_inf:
    with record_function("model_inf"):
        # single forward pass
        for _ in tqdm.tqdm(range(10)):
            model = model.eval()
            X = X.cuda()
            prediction = model(X)


# get table
print(
    prof_train.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=10
    )
)
print(
    prof_inf.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=10
    )
)
