#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(n,sizeof(type)) //same as malloc but set allocated memory to 0


int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;

static char *line = NULL;
static int max_line_len;
int *data_types;
int l;
int max_index;

void construct_data_types(FILE *fp);

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	
	for (int i = 0; i < data_types[0]; i++) { readline(input); }
	while(readline(input) != NULL)
	{
		int i = 0;
		int int_val1, int_val2;
		double target_label, predict_label;
		double dbl_val1, dbl_val2, dbl_val3, dbl_val4;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			
			if(idx == NULL)
				break;
			
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
			{
				val = strtok(NULL," \t\n");
				if(val == NULL){break;}
				exit_input_error(total+1);
			} else {
				inst_max_index = x[i].index;
			}

			errno = 0;
			if(data_types[0] != 0) {
				switch(data_types[x[i].index]) //look at the type of the variable
				{
					case -1:
						exit_input_error(1);
						break;
					case QUANT:
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						(x[i].value).quant = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						break;
					case DICH:
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						(x[i].value).dich = *val;
	//					if(val+sizeof(char) != NULL)      //TODO vérifier que ça marche bien comme attendu 
	//						exit_input_error(total+1);
						break;
					case ORD:
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						(x[i].value).ord = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						break;
					case C_CIRC:
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						(x[i].value).c_circ = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						break;
					case D_CIRC:
						val = strtok(NULL,",");
						if(val == NULL)
							break;
						int_val1 = (int)strtol(val,&endptr,10); //TODO qu'est-ce que le 10 ??
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						int_val2 = (int)strtol(val,&endptr,10); //TODO qu'est-ce que le 10 ??
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						
						(x[i].value).d_circ = (struct int_pair){.first = int_val1, .second = int_val2};
						break;
					case FUZZ:
						val = strtok(NULL,",");
						if(val == NULL)
							break;
						dbl_val1 = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						
						val = strtok(NULL,",");
						if(val == NULL)
							break;
						dbl_val2 = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						val = strtok(NULL,",");
						if(val == NULL)
							break;
						dbl_val3 = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						dbl_val4 = strtod(val,&endptr);
						if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
							exit_input_error(total+1);
						
						(x[i].value).fuzz = (struct fuzzy) {.center = dbl_val1, .left = dbl_val2, .right = dbl_val3, .height = dbl_val4};
						break;
					case MULT:  //TODO à compléter
						(x[i].value).mult = (uint32_t) 0;
						break;
					case NOM:
						val = strtok(NULL," \t\n");
						if(val == NULL)
							break;
						(x[i].value).nom = val;
						break;
				}
			} else {
				val = strtok(NULL," \t\n");
				if(val == NULL)
					break;
				(x[i].value).quant = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(total+1);
			}
			
			++i;
		}
		x[i].index = -1;
		
		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model,data_types,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(model,data_types, x);
			fprintf(output,"%g\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	if(predict_probability)
		free(prob_estimates);
	free(data_types);
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-2)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	construct_data_types(input);

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(data_types, argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	predict(input,output);
	svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}

void construct_data_types(FILE *fp)
{
	int num_line, i_min, i_max, i;
	char *endptr;
	char *label, *idx_min, *idx_max;
	if(readline(fp)!=NULL)
	{
		char *pp = strtok(line," \t");
		if(strcmp(pp,"TYPES")==0)
		{
			label = strtok(NULL," \t\n");
			if(label == NULL)
				exit_input_error(1);
			max_index = (int) strtol(label,&endptr,10);  //TODO qu'est-ce que le 10 ??
			if(endptr == label || *endptr != '\0')
				exit_input_error(1);
			data_types = Calloc(int, max_index+1);
			data_types[0] = 1;
		} else {
			data_types = Calloc(int,1);
			rewind(fp);
		}
	}
	
	num_line = 1;
	l = 0;
	while(readline(fp)!=NULL)
	{
		num_line++;
		char *pp = strtok(line," \t");
		if(strcmp(pp,"TYPES")==0)
		{
			data_types[0]++;
			label = strtok(NULL," \t\n");
			while(1){
				idx_min = strtok(NULL,">");
				if(idx_min == NULL)
					break;
				idx_max = strtok(NULL,", \t\n");
				if(idx_max == NULL)
					break;
				errno = 0;
				i_min = (int) strtol(idx_min,&endptr,10);  //TODO qu'est-ce que le 10 ??
				if(endptr == idx_min || errno != 0 || *endptr != '\0')
					exit_input_error(num_line+1);
				i_max = (int) strtol(idx_max,&endptr,10);  //TODO qu'est-ce que le 10 ??
				if(endptr == idx_max || errno != 0 || *endptr != '\0')
					exit_input_error(num_line+1);
				for (i = i_min; i <= i_max; i++)
				{
					data_types[i] = Types_to_int(label);
				}
			}
		} else {
			l++;
		}
	}
	rewind(fp);
}