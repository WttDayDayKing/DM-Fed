import logging
import numpy as np

def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            logging.info(f"outputs:{outputs}")
            _, predicted = torch.max(outputs.data, 1)
            logging.info(f"predicted:{predicted}")
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred