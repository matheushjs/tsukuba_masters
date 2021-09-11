require(tseriesChaos);
require(rnn);
require(viridis)

# Receives the dataset, which is a matrix with rows (x1, x2, x3, y)
# And receives the query point (x1, x2, x3)
dwnn = function(dataset, query, sigma=0.5){
	yIdx   = ncol(dataset);
	nSamps = nrow(dataset);
	X = as.matrix(dataset[,1:(yIdx-1)], ncol=1);
	Y = as.matrix(dataset[,yIdx], ncol=1);

	if(!is.matrix(query))
		query = matrix(query, nrow=1)

	# Each column of squareDist and weights are relative to each query
	squareDist = apply(query, 1, function(row) rowSums(X - row)**2);
	weights = exp(-squareDist / (2 * sigma**2));

	result = (t(weights) %*% Y) / colSums(weights);
	return(result);
}

report = function(correct, obtained){
	#squares1 = (correct - obtained)**2
	#squares2 = (correct - mean(correct))**2
	#naCount = sum(is.na(squares1));
	#nmse = sum(squares1) / sum(squares2);
	#cat("NMSE: ", nmse, "\tNA samples: ", naCount, "\n");
	#return(nmse);

	rmse = sqrt(mean((correct*4.996839 - obtained*4.996839)**2));
	cat("RMSE: ", rmse, "\n");
	return(rmse);
}

predict.dwnn = function(emb, train.size=0.7){
	m = ncol(emb);

	beginTest = floor(nrow(emb) * train.size);
	train = emb[1:(beginTest-1),];
	test  = emb[beginTest:nrow(emb),];

	# Predict without replacement
	bestSigma = 0.037;
	
	Y = dwnn(train, test[,1:(m-1)], sigma=bestSigma);
	plot(test[,m][1:300], pch=19, xlab="Time", ylab="Stage (m)");
	lines(Y[1:300], col=2, lwd=2);
	rmse = report(test[,m], Y);
	savePlot("dwnn-no-replacement.png");
	return(rmse);

	# Predict with replacement
	#buffer = matrix(0, ncol=m, nrow=nrow(test));
	#buffer[1,] = test[1,];
	#buffer[1,m] = dwnn(train, test[1,1:(m-1)], sigma=0.19);
	#for(i in 2:nrow(test)){
	#	query = buffer[i-1,2:m];
	#	y = dwnn(train, query, sigma=0.19);
	#	buffer[i,] = c(query, y);
	#}

	#plot(test[,m], type="l", xlab="Time", ylab="Sunspots");
	#lines(buffer[,m], col=2);
	#report(test[,m], buffer[,m]);
	#savePlot("dwnn-with-replacement.png");
}

predict.chaotic.rnn = function(emb, train.size=0.7){
	m = ncol(emb);

	X = emb[,1:(m-1)];
	Y = emb[,m];

	# Convert to array as desired by the `rnn` package
	X = array(X, dim=c(dim(X), 1));
	Y = array(Y, dim=c(length(Y), 1));

	# Separate in train / test
	beginTest = floor(nrow(emb) * train.size);
	trainX = X[1:(beginTest-1),,,drop=F];
	trainY = Y[1:(beginTest-1),,drop=F];
	testX  = X[beginTest:nrow(emb),,,drop=F];
	testY  = Y[beginTest:nrow(emb),,drop=F];

	model <- trainr(Y=trainY,
		X=trainX,
		learningrate   = 0.05,
		# learningrate   = 0.04,   # For m=8, d=17
		hidden_dim     = 18,
		batch_size     = 100,
		numepochs      = 1000,
		momentum       = 0.1,
		use_bias       = TRUE,
		seq_to_seq_unsync=TRUE);
	
#	model <- trainr(Y=trainY,
#		X=trainX,
#		learningrate   = 0.06,
#		hidden_dim     = 30,
#		batch_size     = 100,
#		numepochs      = 1000,
#		seq_to_seq_unsync=TRUE);

	# predict
	preds <- predictr(model, testX);

	plot(testY, pch=19, xlab="Time", ylab="Rainflow");
	lines(preds[,1], col=2, lwd=3, lty=3);
	rmse = report(testY, preds[,1]);
	savePlot("chaotic-rnn.png");

	return(rmse);
}


rainflow.mv.embedd = function(
	Q, US1, US2, US3, RG1, RG2, RG3, RG4, RG5,
	lags.Q, lags.US1, lags.US2, lags.RG1, lags.RG2, lags.RG3, lags.RG4, lags.RG5
){
	emb.Q = embedd(Q, lags=lags.Q);
	emb.US1 = embedd(US1, lags=lags.US1);
	emb.US2 = embedd(US2, lags=lags.US2);
	emb.US3 = embedd(US3, lags=lags.US3);
	emb.RG1 = embedd(RG1, lags=lags.RG1);
	emb.RG2 = embedd(RG2, lags=lags.RG2);
	emb.RG3 = embedd(RG3, lags=lags.RG3);
	emb.RG4 = embedd(RG4, lags=lags.RG4);
	emb.RG5 = embedd(RG5, lags=lags.RG5);

	emb.all = list(emb.Q,emb.US1,emb.US2,emb.US3,emb.RG1,emb.RG2,emb.RG3,emb.RG4,emb.RG5);

	min.acc = 2**32;
	for(i in length(emb.all)){
		local.emb = emb.all[[i]];
		
		min.acc = min(min.acc, nrow(local.emb));
	}

	for(i in length(emb.all)){
		local.emb = emb.all[[i]];

		len = nrow(local.emb);

		emb.all[[i]] = local.emb[(len-min.acc + 1):len,];
	}

	full.emb = cbind(emb.all[[9]], emb.all[[8]], emb.all[[7]], emb.all[[6]], emb.all[[5]], emb.all[[4]], emb.all[[3]], emb.all[[2]], emb.all[[1]]);
	return(full.emb);
}

graphics.off();
dev.new(width=14, height=6);

df1 = read.csv("Train1.csv")
df2 = read.csv("Train2.csv")
df3 = read.csv("Test.csv")

test.df = rbind(df1, df2, df3);

series = test.df$Q;
cat("Normalization factor: ", max(abs(series)), "\n");
series = series / max(abs(series));

# Take care here
# lags = c(0,34,34*2,34*3) will feed 34 first to the RNN, and 34*3 lastly (more near the RNN output)

emb = embedd(series, lags=c(4,3,2,1));
#emb = embedd(series, lags=c(0,3,6));

#rmse1 = predict.dwnn(emb);
rmse2 = predict.chaotic.rnn(emb);
#print(c(rmse1, rmse2));



#A = emb[,1:(ncol(emb - 1))];
#B = emb[,ncol(emb)];
#prev = df$mean[1:(length(df$mean)-1)];
#prev = rev(rev(prev)[1:length(B)]);
#emb = cbind(A, prev, B);


